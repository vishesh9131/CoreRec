# Top 5 Quick Reference

Quick lookup for the best methods to use.

## Collaborative Filtering - Top 5

### 1. TwoTower ⭐ (Modern Standard)
```python
from corerec.engines import unionized
model = unionized.TwoTower(embedding_dim=256)
```
**Use when:** Large catalog, real-time serving, first stage retrieval  
**Like:** YouTube, Netflix architecture  
**Speed:** Very fast with vector index  

### 2. SAR (Simple & Fast)
```python
model = unionized.SAR()
```
**Use when:** Quick baseline, no ML infrastructure needed  
**Like:** Item-item collaborative filtering  
**Speed:** Fast, no training  

### 3. LightGCN (Graph-Based)
```python
model = unionized.LightGCN(embedding_dim=128)
```
**Use when:** Social network, user connections matter  
**Like:** Graph neural network approach  
**Speed:** Medium, depends on graph size  

### 4. NCF (Neural Collaborative)
```python
model = unionized.NCF(embedding_dim=64)
```
**Use when:** Learning user-item patterns, interpretability  
**Like:** Neural matrix factorization  
**Speed:** Fast training  

### 5. FastRecommender (Prototyping)
```python
model = unionized.FastRecommender()
```
**Use when:** Rapid prototyping, education, demos  
**Like:** FastAI embedding approach  
**Speed:** Very fast  

---

## Content-Based - Top 5

### 1. TFIDFRecommender (Classic)
```python
from corerec.engines import content
model = content.TFIDFRecommender()
```
**Use when:** Text-based items, no deep learning  
**Like:** Classic information retrieval  
**Speed:** Very fast  

### 2. YoutubeDNN ⭐ (Industry Standard)
```python
model = content.YoutubeDNN(embedding_dim=256)
```
**Use when:** Large-scale production, multi-stage pipeline  
**Like:** YouTube's actual system  
**Speed:** Fast with proper setup  

### 3. DSSM (Semantic Matching)
```python
model = content.DSSM(embedding_dim=128)
```
**Use when:** Query-document matching, semantic understanding  
**Like:** Microsoft's search engine  
**Speed:** Medium  

### 4. BERT4Rec ⭐ (Sequential)
```python
model = content.BERT4Rec(hidden_dim=256)
```
**Use when:** Sequential behavior, time-aware recommendations  
**Like:** BERT for recommendation  
**Speed:** Slower (transformer), high quality  

### 5. Word2VecRecommender (Embeddings)
```python
model = content.Word2VecRecommender(vector_size=100)
```
**Use when:** Item-item similarity, embedding-based  
**Like:** Word2Vec for items  
**Speed:** Fast  

---

## Decision Matrix

### By Scale

| Scale | Collaborative | Content-Based |
|-------|--------------|---------------|
| Small (<10K items) | SAR, NCF | TFIDFRecommender |
| Medium (10K-1M) | LightGCN, NCF | Word2Vec, DSSM |
| Large (>1M) | **TwoTower** | **YoutubeDNN** |

### By Use Case

| Use Case | Collaborative | Content-Based |
|----------|--------------|---------------|
| Real-time serving | **TwoTower** | **YoutubeDNN** |
| Quick baseline | **SAR** | **TFIDFRecommender** |
| Social network | **LightGCN** | - |
| Sequential behavior | - | **BERT4Rec** |
| Item similarity | NCF | **Word2Vec** |
| Semantic matching | - | **DSSM** |

### By Infrastructure

| Infrastructure | Collaborative | Content-Based |
|----------------|--------------|---------------|
| No ML (CPU only) | **SAR** | **TFIDFRecommender** |
| Basic ML | NCF, FastRecommender | Word2Vec, DSSM |
| Full DL (GPU) | **TwoTower**, LightGCN | **YoutubeDNN**, BERT4Rec |

---

## When to Use Sandbox

Use sandbox methods when:
- ✅ Specific algorithm needed (e.g., DeepFM, DCN)
- ✅ Research/experimentation
- ✅ Willing to test thoroughly
- ❌ **Don't** use for production without testing

```python
from corerec.sandbox.collaborative import DeepFM, SASRec
from corerec.sandbox.content_based import CNN, Transformer

# Check what's available
from corerec.engines import unionized
print(unionized.sandbox.list_available())
```

---

## Performance Comparison

Relative speed (higher = faster):

### Collaborative
```
SAR                ████████████ (12/10 - no training)
FastRecommender    ██████████   (10/10)
NCF                ████████     (8/10)
TwoTower           ████████     (8/10 - after index built)
LightGCN           ██████       (6/10 - depends on graph)
```

### Content-Based
```
TFIDFRecommender   ████████████ (12/10)
Word2Vec           ██████████   (10/10)
YoutubeDNN         ████████     (8/10)
DSSM               ██████       (6/10)
BERT4Rec           ████         (4/10 - high quality though)
```

---

## Code Templates

### Template 1: Quick Baseline
```python
from corerec.engines import unionized
import numpy as np

# Data
user_ids = [...]
item_ids = [...]
interactions = np.array([...])

# Fast baseline
model = unionized.SAR()
model.fit(user_ids, item_ids, interactions)
recs = model.recommend(user_id, top_k=10)
```

### Template 2: Production Scale
```python
from corerec.engines import unionized
from corerec.retrieval.vector_store import create_index

# Two-Tower model
model = unionized.TwoTower(embedding_dim=256)
model.fit(user_ids, item_ids, interactions)

# FAISS index for speed
item_embs = model.get_item_embeddings()
index = create_index("faiss", dim=256, index_type="hnsw")
index.add(item_embs, item_ids)
index.save("items.faiss")

# Fast retrieval
user_emb = model.get_user_embedding(user_id)
scores, candidates = index.search(user_emb, k=1000)
```

### Template 3: Sequential
```python
from corerec.engines import content

# BERT4Rec for sequences
model = content.BERT4Rec(
    hidden_dim=256,
    num_layers=4,
    num_heads=8
)

model.fit(user_ids, item_ids, interactions)
next_items = model.recommend(user_id, top_k=10)
```

---

## FAQ

**Q: Which is the "best" method?**  
A: Depends on use case. For most: TwoTower (collab) + YoutubeDNN (content).

**Q: Can I use multiple methods?**  
A: Yes! Use TwoTower for retrieval, then another for ranking.

**Q: What about hybrid (collab + content)?**  
A: Use multi-modal fusion or ensemble both types.

**Q: Are these really production-ready?**  
A: Yes. All Top 5 are battle-tested (some at FAANG companies).

**Q: What if I need a specific algorithm?**  
A: Check sandbox. Most research algorithms are there.

---

## Cheat Sheet

| Need | Use This |
|------|----------|
| Fastest baseline | SAR or TFIDFRecommender |
| Best quality | TwoTower + BERT4Rec |
| Production scale | TwoTower + YoutubeDNN |
| No ML needed | SAR + TFIDFRecommender |
| Social network | LightGCN |
| Sequential | BERT4Rec |
| Text items | TFIDFRecommender or Word2Vec |
| Prototyping | FastRecommender |
| Semantic search | DSSM |

---

**Still unsure?** Start with:
- Collaborative: `unionized.SAR()` (baseline) → `unionized.TwoTower()` (production)
- Content: `content.TFIDFRecommender()` (baseline) → `content.YoutubeDNN()` (production)

Then iterate based on results.

