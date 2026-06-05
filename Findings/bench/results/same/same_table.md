| Family | Framework | Model | NDCG@10 | Recall@10 | HitRate@10 | Fit (s) | Mem (MB) | Device |
|---|---|---|---|---|---|---|---|---|
| ALS | implicit | ALS | 0.0319 | 0.0526 | 0.1400 | 0.9361 | 467.6000 | cpu |
| BPR | cornac | BPR | 0.0453 | 0.0790 | 0.1400 | 0.2171 | 467.4000 | cpu |
| BPR | lightfm | BPR | 0.0110 | 0.0209 | 0.0600 | 1.4943 | 469.0000 | cpu |
| BPR | implicit | BPR | 0.0020 | 0.0024 | 0.0200 | 0.5522 | 468.9000 | cpu |
| DCN | corerec | DCN | 0.0117 | 0.0133 | 0.0500 | 77.4879 | 775.6000 | cpu |
| DeepFM | corerec | DeepFM | 0.0176 | 0.0317 | 0.0500 | 306.0591 | 802.5000 | cpu |
| GNN/NGCF | recbole | NGCF | 0.0125 | 0.0193 | 0.0614 | 111.1316 | 2188.2000 | cuda |
| LightGCN | corerec | LightGCN | 0.0441 | 0.0807 | 0.1400 | 14.1250 | 1476.0000 | cuda |
| LightGCN | recbole | LightGCN | 0.0245 | 0.0377 | 0.1100 | 108.0063 | 2106.4000 | cuda |
| MF | cornac | MF | 0.0047 | 0.0100 | 0.0200 | 0.0507 | 468.6000 | cpu |
| MF | surprise | SVD | 0.0025 | 0.0060 | 0.0200 | 0.4447 | 467.8000 | cpu |
| MF | cornac | WMF | 0.0013 | 0.0020 | 0.0100 | 5.7980 | 871.2000 | cpu |
| Neighborhood-CF | corerec | SAR | 0.0111 | 0.0171 | 0.0400 | 0.4098 | 944.2000 | cpu |
| Neighborhood-CF | implicit | ItemKNN | 0.0020 | 0.0050 | 0.0100 | 0.0192 | 469.8000 | cpu |
| Neighborhood-CF | cornac | ItemKNN | 0.0000 | 0.0000 | 0.0000 | 0.2973 | 467.9000 | cpu |
| Neighborhood-CF | cornac | UserKNN | 0.0000 | 0.0000 | 0.0000 | 0.5751 | 604.1000 | cpu |
| NeuMF/NCF | recbole | NeuMF | 0.0227 | 0.0357 | 0.1079 | 57.5225 | 2072.6000 | cuda |
| NeuMF/NCF | cornac | GMF | 0.0147 | 0.0308 | 0.0500 | 213.1423 | 840.4000 | cpu |
| NeuMF/NCF | cornac | NeuMF | 0.0136 | 0.0285 | 0.0700 | 324.6786 | 1561.1000 | cpu |
| NeuMF/NCF | corerec | NCF | 0.0076 | 0.0153 | 0.0400 | 127.8962 | 776.3000 | cpu |
| VAE | cornac | VAECF | 0.0445 | 0.0840 | 0.1500 | 8.2169 | 945.3000 | cpu |
| WARP | lightfm | WARP | 0.0182 | 0.0344 | 0.0900 | 1.6154 | 468.9000 | cpu |
