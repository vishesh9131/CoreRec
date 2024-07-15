from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, ModuleList, Dropout, LayerNorm
import torch

class GraphTransformerV2(Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True, dropout=0.1):
        super(GraphTransformerV2, self).__init__()
        self.num_weights = num_weights
        self.use_weights = use_weights
        self.input_linear = Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = Linear(d_model, input_dim)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        if self.use_weights:
            self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])

    def forward(self, x, weights=None):
        x = x.float()
        if self.use_weights:
            if weights is not None:
                weighted_x = torch.zeros_like(x)
                for i, weight in enumerate(weights):
                    weighted_x += self.weight_linears[i](x) * weight
                x = weighted_x
            else:
                x = self.input_linear(x)
        else:
            x = self.input_linear(x)
        
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        x = self.dropout(x)
        return x