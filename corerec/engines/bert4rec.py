"""
BERT4Rec: bidirectional sequential recommender using transformers.

Unlike SASRec (causal), this uses bidirectional attention like BERT.
Good for capturing complex sequential patterns.

Key difference from original implementations: 
we focus on practical usage over paper replication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math
import logging

from corerec.api.base_recommender import BaseRecommender


class TransformerBlock(nn.Module):
    """Single transformer layer with multi-head attention + FFN."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] bool tensor (True = padding)
        """
        # self attention with pre-norm
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        
        # feed forward with pre-norm
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x


class BERT4RecModel(nn.Module):
    """
    Bidirectional transformer for sequence modeling.
    
    Unlike causal models, can look at full context when predicting masked items.
    """
    
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 max_len: int = 200,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # item embeddings (vocab_size includes padding=0 and mask token)
        self.item_emb = nn.Embedding(vocab_size + 2, hidden_dim, padding_idx=0)
        self.mask_token_id = vocab_size + 1
        
        # positional embeddings
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # output projection to vocab
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Small init to avoid instability."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq: [batch, seq_len] with item indices (0 = padding, mask_token_id = masked)
        mask: [batch, seq_len] bool (True = padding)
        
        Returns: logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = seq.size()
        
        # embed items
        x = self.item_emb(seq)  # [batch, seq_len, hidden]
        
        # add positional encoding
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_emb(positions)
        
        x = self.dropout(x)
        
        # pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_final(x)
        
        # project to vocab
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return logits
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Return embeddings for all items (excluding padding/mask)."""
        return self.item_emb.weight[1:self.vocab_size+1]


class BERT4Rec(BaseRecommender):
    """
    BERT-style sequential recommender.
    
    Training: randomly mask items in sequences, predict masked items.
    Inference: append mask token, predict it.
    """
    
    def __init__(self,
                 name: str = "BERT4Rec",
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 max_len: int = 200,
                 dropout: float = 0.1,
                 mask_prob: float = 0.15,
                 learning_rate: float = 1e-4,
                 batch_size: int = 64,
                 num_epochs: int = 10,
                 device: Optional[torch.device] = None,
                 verbose: bool = True):
        super().__init__()
        
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.mask_prob = mask_prob
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.verbose = verbose
        
        self.log = logging.getLogger(self.name)
        if verbose:
            self.log.setLevel(logging.INFO)
        
        # initialized during fit
        self.model = None
        self.optimizer = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_seqs = {}
        self.is_fitted = False
    
    def fit(self, user_ids: List, item_ids: List, interactions: np.ndarray):
        """
        Train BERT4Rec model.
        
        user_ids: list of user IDs
        item_ids: list of item IDs
        interactions: [n_users, n_items] binary matrix
        """
        self.log.info(f"Training {self.name}...")
        
        # build item vocab
        self.item_to_idx = {iid: idx+1 for idx, iid in enumerate(item_ids)}  # 0 reserved for padding
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
        
        vocab_size = len(item_ids)
        
        # build user sequences
        self.user_seqs = {}
        train_seqs = []
        
        for u_idx, user_id in enumerate(user_ids):
            item_indices = interactions[u_idx].nonzero()[0]
            if len(item_indices) < 2:
                continue  # need at least 2 items
            
            seq = [self.item_to_idx[item_ids[i]] for i in item_indices]
            self.user_seqs[user_id] = seq
            
            # create training samples by sliding window
            for start in range(0, len(seq), self.max_len // 2):
                subseq = seq[start:start + self.max_len]
                if len(subseq) >= 2:
                    train_seqs.append(subseq)
        
        self.log.info(f"Created {len(train_seqs)} training sequences from {len(self.user_seqs)} users")
        
        if len(train_seqs) == 0:
            self.log.warning("No valid sequences found")
            return self
        
        # init model
        self.model = BERT4RecModel(
            vocab_size=vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_len=self.max_len,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            
            np.random.shuffle(train_seqs)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_seqs), self.batch_size):
                batch_seqs = train_seqs[i:i+self.batch_size]
                
                # prepare batch with masking
                input_seqs, target_seqs, mask_positions = self._prepare_batch(batch_seqs, vocab_size)
                
                input_seqs = torch.tensor(input_seqs, dtype=torch.long, device=self.device)
                target_seqs = torch.tensor(target_seqs, dtype=torch.long, device=self.device)
                
                # padding mask
                pad_mask = (input_seqs == 0)
                
                self.optimizer.zero_grad()
                
                logits = self.model(input_seqs, pad_mask)  # [batch, seq_len, vocab]
                
                # compute loss only on masked positions
                loss = 0
                n_masked = 0
                for b_idx, positions in enumerate(mask_positions):
                    for pos in positions:
                        loss += F.cross_entropy(
                            logits[b_idx, pos],
                            target_seqs[b_idx, pos],
                            reduction='sum'
                        )
                        n_masked += 1
                
                if n_masked > 0:
                    loss = loss / n_masked
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            
            if self.verbose and (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                self.log.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        self.log.info("Training complete")
        return self
    
    def _prepare_batch(self, seqs: List[List[int]], vocab_size: int) -> Tuple:
        """
        Mask random items in sequences for training.
        
        Returns: (input_seqs, target_seqs, mask_positions)
        """
        max_seq_len = max(len(s) for s in seqs)
        batch_size = len(seqs)
        
        input_seqs = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        target_seqs = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_positions = []
        
        mask_token = vocab_size + 1
        
        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            
            # randomly mask some positions
            n_mask = max(1, int(seq_len * self.mask_prob))
            mask_pos = np.random.choice(seq_len, size=n_mask, replace=False)
            mask_positions.append(mask_pos.tolist())
            
            for j, item_idx in enumerate(seq):
                if j in mask_pos:
                    # 80% mask, 10% random, 10% keep
                    r = np.random.random()
                    if r < 0.8:
                        input_seqs[i, j] = mask_token
                    elif r < 0.9:
                        input_seqs[i, j] = np.random.randint(1, vocab_size + 1)
                    else:
                        input_seqs[i, j] = item_idx
                    
                    target_seqs[i, j] = item_idx
                else:
                    input_seqs[i, j] = item_idx
                    target_seqs[i, j] = item_idx  # ignored in loss but kept for shape
        
        return input_seqs, target_seqs, mask_positions
    
    def recommend(self, user_id: Any, top_k: int = 10, exclude_seen: bool = True) -> List[Any]:
        """
        Generate recommendations by appending mask token and predicting.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if user_id not in self.user_seqs:
            self.log.warning(f"Unknown user: {user_id}")
            return []
        
        seq = self.user_seqs[user_id][-self.max_len:]  # use recent history
        
        # append mask token
        input_seq = seq + [self.model.mask_token_id]
        input_seq = torch.tensor([input_seq], dtype=torch.long, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_seq)  # [1, seq_len, vocab]
            last_logits = logits[0, -1, :]  # predict the mask token
            
            scores = F.softmax(last_logits, dim=-1).cpu().numpy()
        
        # exclude padding
        scores[0] = -np.inf
        
        # exclude seen items
        if exclude_seen:
            for idx in seq:
                if 0 < idx < len(scores):
                    scores[idx] = -np.inf
        
        # top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = [self.idx_to_item.get(idx) for idx in top_indices if idx in self.idx_to_item]
        
        return [r for r in recommendations if r is not None]

