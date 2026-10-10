# Removed Models

CoreRec 0.7.0 removed the experimental `corerec.sandbox` package. These 51
models lived there, and their tutorials went with it: their imports no longer
work. Each row names the closest model that ships today.

To keep using one of these architectures, write it as a plain PyTorch module
and wrap it in `corerec.nn.Recommender`, which supplies training, evaluation,
saving, serving and ONNX export (see the Custom Models guide).

| Removed | What it was | Closest in CoreRec |
|---|---|---|
| A2SVD | Adaptive Singular Value Decomposition | ALS |
| AFM | Attentional Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| ALS | Alternating Least Squares | ALS |
| AutoFI | Automatic Feature Interaction | [DeepFM](deepfm_tutorial.md) |
| AutoInt | Automatic Feature Interaction Learning via Self-Attention | [DeepFM](deepfm_tutorial.md) |
| BiVAE | Bilateral Variational Autoencoder | MultVAE |
| BPR | Bayesian Personalized Ranking | ALS, or `corerec.nn.MatrixFactorization` with `loss="bpr"` |
| BPRMF | Bayesian Personalized Ranking Matrix Factorization | ALS, or `corerec.nn.MatrixFactorization` with `loss="bpr"` |
| BST | Behavior Sequence Transformer | [SASRec](sasrec_tutorial.md) |
| Caser | Convolutional Sequence Embedding Recommendation | [SASRec](sasrec_tutorial.md) |
| DCN_base | Deep & Cross Network Base | [DCN](dcn_tutorial.md) |
| DeepCrossing | Deep Crossing Network | [DCN](dcn_tutorial.md) |
| DeepFM_base | DeepFM Base | [DeepFM](deepfm_tutorial.md) |
| DeepRec | Deep Recommender | MultiDAE |
| DIEN | Deep Interest Evolution Network | [SASRec](sasrec_tutorial.md) |
| DIFM | Dual Input Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| DIN | Deep Interest Network | [SASRec](sasrec_tutorial.md) |
| DLRM | Deep Learning Recommendation Model | [DCN](dcn_tutorial.md) |
| ENSFM | Ensemble Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| ESCMM | Entire Space Cross Multi-Task Model | [DeepFM](deepfm_tutorial.md) per task (no multi-task model) |
| ESMM | Entire Space Multi-Task Model | [DeepFM](deepfm_tutorial.md) per task (no multi-task model) |
| FFM | Field-aware Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| FGCNN | Feature Generation with CNN | [DeepFM](deepfm_tutorial.md) |
| Fibinet | Feature Importance and Bilinear feature Interaction | [DeepFM](deepfm_tutorial.md) |
| FLEN | Feature-aware Local Encoding Network | [DeepFM](deepfm_tutorial.md) |
| FM_Base | Factorization Machine Base | [DeepFM](deepfm_tutorial.md) |
| FM | Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| GAN | Generative Adversarial Network | MultiDAE |
| GateNet | Gating Network | [DeepFM](deepfm_tutorial.md) |
| GeoIMC | Geographic Inductive Matrix Completion | ALS |
| GNN_base | Graph Neural Network Base | [LightGCN](lightgcn_tutorial.md) |
| GRU-CF | GRU for Collaborative Filtering | [SASRec](sasrec_tutorial.md) |
| LightGCN_Base | LightGCN Base | [LightGCN](lightgcn_tutorial.md) |
| MatrixFactorization | Matrix Factorization | ALS |
| MF_Base | MF Base | ALS |
| MIND_Content | MIND for Content Filtering | [TwoTower](two_tower_tutorial.md) |
| MMoE | Multi-gate Mixture-of-Experts | [DeepFM](deepfm_tutorial.md) per task (no multi-task model) |
| NextItNet | Next Item Net | [SASRec](sasrec_tutorial.md) |
| NFM | Neural Factorization Machine | [DeepFM](deepfm_tutorial.md) |
| PLE | Progressive Layered Extraction | [DeepFM](deepfm_tutorial.md) per task (no multi-task model) |
| PNN | Product-based Neural Network | [DeepFM](deepfm_tutorial.md) |
| RBM | Restricted Boltzmann Machine | MultiDAE |
| RLRMC | Reinforcement Learning for Recommendation | ALS |
| SLiRec | Sequential List Recommendation | [SASRec](sasrec_tutorial.md) |
| SUM | Sequential User Model | [SASRec](sasrec_tutorial.md) |
| SVD | Singular Value Decomposition | ALS |
| TDM | Tree-based Deep Model | [TwoTower](two_tower_tutorial.md) |
| UserBased | User-Based Collaborative Filtering | UserKNN |
| VMF | Von Mises-Fisher Distribution | MultVAE |
| WideDeep | Wide & Deep Learning | [DCN](dcn_tutorial.md) |
| YouTubeDNN | YouTube Deep Neural Network | [TwoTower](two_tower_tutorial.md) |
