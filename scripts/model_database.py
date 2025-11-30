# CoreRec Model Information Database
# Comprehensive, model-specific details for documentation generation

MODEL_DATABASE = {
    # ========== CORE ENGINE MODELS ==========
    "DCN": {
        "full_name": "Deep & Cross Network",
        "paper": "Wang et al. 2017 - Deep & Cross Network for Ad Click Predictions",
        "intro": "combines explicit feature crossing with deep learning to model both low and high-order feature interactions efficiently",
        "architecture": """**Two Parallel Networks:**

1. **Cross Network**: Learns explicit bounded-degree feature interactions
   - Applies element-wise multiplication at each layer
   - Models feature crossing efficiently: `x_{l+1} = x_0 · x_l^T · w_l + b_l + x_l`
   - Each layer increases polynomial degree by 1

2. **Deep Network**: Learns implicit high-order interactions
   - Standard fully-connected neural network
   - Captures complex non-linear patterns
   
3. **Combination Layer**: Concatenates outputs and applies final transformation

**Flow:**
```
Features → [Cross Network] \
                            → Concat → Dense → Prediction
Features → [Deep Network]  /
```""",
        "math": """**Cross Network Layer l:**
```
x_{l+1} = x_0 · x_l^T · w_l + b_l + x_l
```
where:
- `x_0` is input feature vector  
- `x_l` is output of layer l
- `w_l, b_l` are learnable parameters
- Complexity: O(d) per layer

**Deep Network Layer l:**
```
h_{l+1} = ReLU(W_l · h_l + b_l)
```

**Final Prediction:**
```
y = σ([x_cross; h_deep] · w_out + b_out)
```""",
        "use_cases": """✅ **Excellent For:**
- Feature-rich datasets (user demographics, item attributes, context)
- Ad click prediction and CTR estimation
- When explicit feature crossing matters (e.g., age × gender, category × price)
- Datasets with 100K-10M interactions
- E-commerce product recommendations

❌ **Not Ideal For:**
- Pure collaborative filtering (no features) → use Matrix Factorization
- Sequential/temporal patterns → use SASRec or LSTM
- Graph-structured data → use GNN
- Very large scale (>100M interactions) → use simpler FM""",
        "best_practices": """1. **Feature Engineering**: DCN shines with good features - invest time here
2. **Cross Layers**: 2-3 layers sufficient (more = overfitting risk)
3. **Deep Layers**: Start with [128, 64], add depth for complex datasets
4. **Embedding Dimension**: 32-128 based on feature cardinality
5. **Regularization**: Use dropout (0.2-0.3) and L2 (1e-5)
6. **Normalization**: Normalize continuous features to [0,1] or [-1,1]
7. **Learning Rate**: 0.001 with decay works well
8. **Batch Size**: 256-512 for stability""",
        "dataset": "movielens-100k",
        "init_params": """    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64],
    dropout=0.2,""",
    },
    "DeepFM": {
        "full_name": "Factorization Machine with Deep Learning",
        "paper": "Guo et al. 2017 - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction",
        "intro": "combines Factorization Machines for 2nd-order interactions with Deep Neural Networks for high-order interactions, sharing the same embedding layer",
        "architecture": """**Shared Embedding + Two Components:**

1. **FM Component**: Models 2nd-order feature interactions
   - Linear part: first-order features
   - Interaction part: pairwise interactions via dot products
   - Same embeddings as DNN (parameter sharing!)

2. **Deep Component**: Models high-order interactions  
   - Stacked fully-connected layers
   - Learns complex non-linear combinations

3. **Shared Embedding Layer**: Both components use same embeddings
   - Reduces parameters
   - FM improves DNN's embedding learning

**Flow:**
```
Raw Features → Embedding Layer → [FM Component] \
                                               → Sum → Prediction
                      ↓            [DNN Component]/
                  (shared)
```""",
        "math": """**FM Component:**
```
y_FM = w_0 + Σ_i w_i·x_i + Σ_i Σ_j <v_i, v_j>·x_i·x_j
```
where `<v_i, v_j>` is dot product of embeddings

**DNN Component:**
```
a^(0) = [e_1, e_2, ..., e_m]  # concatenated embeddings
a^(l+1) = σ(W^(l) · a^(l) + b^(l))
y_DNN = W^(out) · a^(L) + b^(out)
```

**Final Prediction:**
```
ŷ = sigmoid(y_FM + y_DNN)
```""",
        "use_cases": """✅ **Ideal For:**
- CTR prediction in advertising
- Sparse high-dimensional features (categorical data)
- Click-through rate estimation  
- App recommendation
- When you need both low-order and high-order interactions
- Large-scale industrial applications

❌ **Avoid When:**
- Dense feature spaces (images, text)
- Sequential patterns dominate → use RNN/Transformer
- Graph structure important → use GNN  
- Very small datasets (<10K) → use simpler models""",
        "best_practices": """1. **Embedding Dimension**: 8-32 for sparse data, 64-128 for rich features
2. **DNN Architecture**: [256, 128, 64] for complex patterns
3. **Dropout**: 0.3-0.5 to prevent overfitting
4. **Batch Normalization**: Apply after each dense layer
5. **Negative Sampling**: Essential for implicit feedback
6. **Batch Size**: 512-2048 for large datasets
7. **Optimizer**: Adam with lr=0.001
8. **Feature Hashing**: Use for very high-cardinality features""",
        "dataset": "criteo-mini",
        "init_params": """    embedding_dim=32,
    deep_layers=[256, 128, 64],
    dropout=0.3,
    use_bn=True,""",
    },
    "GNNRec": {
        "full_name": "Graph Neural Network Recommender",
        "paper": "Hamilton et al. 2017 - Inductive Representation Learning on Large Graphs",
        "intro": "leverages graph neural networks to learn user and item representations by aggregating information from neighboring nodes in the interaction graph",
        "architecture": """**Graph-Based Learning:**

1. **Graph Construction**: 
   - Nodes: Users and Items
   - Edges: Interactions (ratings, clicks, purchases)
   - Bipartite graph structure

2. **Message Passing**: L layers of neighborhood aggregation
   - Aggregate neighbor information
   - Update node representations
   - Stack multiple layers for multi-hop propagation

3. **Aggregation Functions**:
   - Mean: `h_v = σ(W · MEAN{h_u : u ∈ N(v)})`
   - Sum: `h_v = σ(W · SUM{h_u : u ∈ N(v)})`
   - Attention: `h_v = σ(Σ α_uv · h_u)`

**Flow:**
```
Graph → Embed → Aggregate^(1) → ... → Aggregate^(L) → Predict
```""",
        "math": """**Message Passing Layer l:**
```
h_v^(l) = σ(W^(l) · AGG({h_u^(l-1) : u ∈ N(v)}) + b^(l))
```

**Mean Aggregator:**
```
AGG = (1/|N(v)|) · Σ_{u ∈ N(v)} h_u^(l-1)
```

**Attention Aggregator:**
```
α_uv = exp(LeakyReLU(a^T [W·h_u || W·h_v])) / Z
h_v = σ(Σ_{u ∈ N(v)} α_uv · W · h_u)
```

**Prediction:**
```
score(u,i) = h_u^(L)^T · h_i^(L)
```""",
        "use_cases": """✅ **Best For:**
- Social networks with user connections (trust, follow, friendship)
- Multi-hop relationships matter (friend-of-friend)
- Cold-start users (can leverage network)
- Heterogeneous graphs (users, items, categories, attributes)
- When graph structure provides signal

❌ **Not For:**
- No graph structure available → use collaborative filtering
- Very sparse graphs (avg degree < 2) → not enough signal
- Real-time constraints (GNN can be slow)
- Pure sequential patterns → use RNN""",
        "best_practices": """1. **Number of Layers**: 2-3 layers optimal (more causes over-smoothing)
2. **Aggregator Choice**: Mean for homogeneous, Attention for heterogeneous
3. **Embedding Dimension**: 64-128 for most graphs
4. **Neighborhood Sampling**: Sample 10-25 neighbors per node (for scalability)
5. **Mini-batch Training**: Use GraphSAINT or cluster-GCN for large graphs
6. **Dropout**: 0.1-0.3 on edges/messages
7. **Skip Connections**: Add for deep networks (>3 layers)
8. **Negative Sampling**: 5-10 negatives per positive edge""",
        "dataset": "epinions",  # has social network
        "init_params": """    embedding_dim=128,
    num_layers=3,
    aggregator='mean',
    dropout=0.1,
    negative_samples=5,""",
    },
    # Continue with more models...
    "MIND": {
        "full_name": "Multi-Interest Network with Dynamic Routing",
        "paper": "Li et al. 2019 - Multi-Interest Network with Dynamic Routing",
        "intro": "captures diverse user interests using a multi-interest extraction layer with capsule networks and dynamic routing",
        "architecture": """**Capsule-Based Multi-Interest Extraction:**

1. **Behavior Embedding**: Embed user's historical items
2. **Multi-Interest Capsules**: Extract K diverse interests via capsule network
3. **Dynamic Routing**: Route item embeddings to interest capsules
4. **Label-Aware Attention**: Attend to relevant interests for target item
5. **Aggregation**: Combine attended interests for prediction

**Key Innovation**: Models users as having multiple interests rather than single preference vector

**Architecture Flow:**
```
Items → Embed → Capsule(K interests) → Label Attention → Predict
                     ↓
              Dynamic Routing (3 iterations)
```""",
        "math": """**Interest Capsule Extraction:**
```
S_j = Σ_i c_ij · û_i|j    # weighted sum
v_j = squash(S_j)         # capsule activation
where û_i|j = W · e_i     # transformed embeddings
```

**Dynamic Routing (3 iterations):**
```
b_ij ← b_ij + û_i|j · v_j     # update routing logits
c_ij = softmax_j(b_ij)         # routing coefficients
```

**Label-Aware Attention:**
```
score_k = softmax(e_target^T · interest_k / √d)
user_vector = Σ_k score_k · interest_k
ŷ = σ(user_vector^T · e_target)
```""",
        "use_cases": """✅ **Perfect For:**
- E-commerce with diverse catalogs (fashion + electronics + books)
- Users with multi-faceted interests
- Capturing interest drift/evolution
- Session-based recommendations with multiple intent
- Personalized diverse recommendations

❌ **Avoid For:**
- Single-domain recommendations (music only, movies only)
- Very sparse data (<50 items per user)
- Users with narrow interests
- Real-time serving (more complex than single-vector models)""",
        "best_practices": """1. **Number of Interests (K)**: 4-8 typical, more for very diverse catalogs
2. **Routing Iterations**: 3 iterations standard (more doesn't help much)
3. **Capsule Dimension**: Same as embedding (64-128)
4. **Sequence Length**: Use 20-100 recent items
5. **Interest Regularization**: Add orthogonality loss to prevent collapse
6. **Auxiliary Losses**: Train each interest capsule independently
7. **Serving**: Cache interests, only attend at inference
8. **Hard Negative Mining**: Sample from different interest clusters""",
        "dataset": "amazon-books",
        "init_params": """    embedding_dim=64,
    num_interests=4,
    routing_iterations=3,
    seq_len=50,
    use_interest_regularization=True,""",
    },
    "NASRec": {
        "full_name": "Neural Architecture Search for Recommendation",
        "paper": "Various NAS papers applied to RecSys",
        "intro": "automatically discovers optimal neural network architectures using reinforcement learning or evolutionary algorithms",
        "architecture": """**Automated Architecture Discovery:**

1. **Search Space Definition**:
   - Operations: Conv, LSTM, Attention, MLP, Pooling
   - Connections: Skip, Residual, Dense
   - Hyperparameters: Hidden sizes, activations

2. **Search Strategy**:
   - Controller: RNN or Evolution Algorithm
   - Proposes candidate architectures
   - Evaluates on validation set
   - Updates based on performance

3. **Architecture Encoding**:
   - Sequence of layer types and configurations
   - Example: [LSTM-64, Attention-32, MLP-128, ...]

**Search Process:**
```
Random → Controller → Candidate Arch → Train → Evaluate
  ↑                                                ↓
  └──────────── Update Controller ←───────────────┘
```""",
        "math": """**Architecture Sampling:**
```
arch ~ Controller(θ)
arch = [layer_1, ..., layer_n]
where layer_i = (type_i, config_i)
```

**Reward Function:**
```
R(arch) = α · Metric(arch) - β · Latency(arch) - γ · Params(arch)
```
Balances accuracy, speed, and model size

**Controller Update (REINFORCE):**
```
∇_θ J = E_{arch~p(·|θ)}[R(arch) · ∇_θ log p(arch|θ)]
θ ← θ + η · ∇_θ J
```""",
        "use_cases": """✅ **Ideal For:**
- Novel domains without established architectures
- Research and experimentation
- When you have significant compute budget (100+ GPU hours)
- Performance-critical applications worth the search cost
- AutoML platforms

❌ **Not For:**
- Quick prototyping (search takes days/weeks)
- Limited compute (<10 GPUs)
- Well-solved domains (just use proven architectures)
- Frequently changing data (search doesn't transfer)
- Production with tight constraints""",
        "best_practices": """1. **Search Budget**: Minimum 100-500 architecture evaluations  
2. **Early Stopping**: Stop bad architectures at epoch 5
3. **Warm Start**: Initialize with known good architectures
4. **Constrained Search**: Limit latency/params to feasible range
5. **Transfer Learning**: Fine-tune found architecture
6. **Multi-Objective**: Use Pareto frontier for accuracy-latency trade-off
7. **Supernet Training**: Train once, search multiple times
8. **Progressive Search**: Start simple, add complexity gradually""",
        "dataset": "movielens-1m",
        "init_params": """    search_space='wide',  # 'narrow', 'wide', 'full'
    search_iterations=200,
    epochs_per_arch=10,
    max_params=5e6,
    max_latency_ms=50,""",
    },
    "SASRec": {
        "full_name": "Self-Attentive Sequential Recommendation",
        "paper": "Kang & McAuley 2018 - Self-Attentive Sequential Recommendation",
        "intro": "uses self-attention mechanism to model sequential user behavior, capturing both short and long-term patterns",
        "architecture": """**Transformer-Based Sequential Modeling:**

1. **Input**: Sequence of user's items [i₁, i₂, ..., iₙ]
2. **Embedding Layer**: Item embeddings + positional encoding
3. **Self-Attention Blocks** (stacked L times):
   - Multi-head self-attention with causal masking
   - Point-wise feed-forward network
   - Layer normalization + Residual connections
4. **Prediction**: Final item representation

**Key Innovation**: Self-attention captures long-range dependencies better than RNN/CNN

**Architecture:**
```
Items → Embed + Position → Self-Attention×L → Predict Next
```""",
        "math": """**Self-Attention:**
```
Q = EW^Q, K = EW^K, V = EW^V
Attention(Q,K,V) = softmax(QK^T/√d_k + M) V
```

**Causal Mask M** (prevent attending to future):
```
M_ij = { 0    if i ≥ j
       {-∞    if i < j
```

**Multi-Head Attention:**
```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)  
MultiHead = Concat(head_1, ..., head_h)W^O
```

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Feed-Forward:**
```
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```""",
        "use_cases": """✅ **Perfect For:**
- Sequential user behavior (browsing, watching, listening)
- Session-based recommendations
- Long sequences (50-200 items)
- Capturing long-range dependencies
- Next-item prediction
- E-commerce, streaming platforms, news

❌ **Not Suitable For:**
- Very short sequences (<5 items)
- Static ratings (no sequence)
- Graph-structured data
- When interpretability is critical
- Memory-constrained systems""",
        "best_practices": """1. **Sequence Length**: 50-200 items optimal (truncate longer)
2. **Attention Blocks**: 2-4 blocks (L=2 often sufficient)
3. **Attention Heads**: 1-4 heads (h=2 works well)
4. **Hidden Dimension**: 50-100 (d model)
5. **Dropout**: 0.2-0.5 for regularization
6. **Positional Encoding**: ESSENTIAL (order matters!)
7. **Learning Rate**: 0.001 with linear warmup (1000 steps)
8. **Batch Size**: Large (128-512) with padding
9. **Negative Sampling**: Sample popular items as negatives""",
        "dataset": "amazon-beauty",
        "init_params": """    max_seq_len=50,
    hidden_dim=50,
    num_blocks=2,
    num_heads=2,
    dropout=0.2,
    use_positional_encoding=True,""",
    },
    "NCF": {
        "full_name": "Neural Collaborative Filtering",
        "paper": "He et al. 2017 - Neural Collaborative Filtering",
        "intro": "replaces traditional matrix factorization's inner product with a neural network to learn complex user-item interactions",
        "architecture": """**Neural Generalization of Matrix Factorization:**

1. **Embedding Layer**: Separate embeddings for GMF and MLP paths
2. **GMF Path** (Generalized Matrix Factorization):
   - Element-wise product of user and item embeddings
   - Mimics MF but with learnable weights
3. **MLP Path**: Multi-layer neural network
   - Concatenates user and item embeddings
   - Learns non-linear interactions
4. **NeuMF Layer**: Combines GMF and MLP outputs

**Key Innovation**: Replaces dot product with flexible neural architecture

**Architecture:**
```
User → [GMF Embedding] → Element-wise × \
                                         → Concat → Dense → Prediction  
Item → [GMF Embedding] → Element-wise × /

User → [MLP Embedding] → \
                          Concat → MLP Layers → /
Item → [MLP Embedding] → /
```""",
        "math": """**GMF Component:**
```
φ_GMF(u,i) = a_out(h^T (p_u ⊙ q_i))
```
where ⊙ is element-wise product

**MLP Component:**
```
z_1 = [p_u; q_i]
z_{l+1} = σ(W_l^T z_l + b_l)
φ_MLP(u,i) = a_out(h^T z_L)
```

**NeuMF (Combined):**
```
ŷ_ui = σ(h^T [φ_GMF(u,i); φ_MLP(u,that)])
```
where ; denotes concatenation""",
        "use_cases": """✅ **Excellent For:**
- Implicit feedback data (clicks, views, purchases)
- Collaborative filtering with non-linear patterns
- When MF is too simple but you want interpretability
- Medium-scale datasets (100K-10M interactions)
- Rating prediction and top-N recommendation

❌ **Not Ideal For:**
- Very sparse data (MF works better)
- Need to incorporate features → use Deep models
- Sequential patterns → use RNN/Transformer
- Very large scale (>50M interactions) → use simpler MF for speed""",
        "best_practices": """1. **Embedding Dimension**: 8-64 typically (smaller than image/NLP)
2. **MLP Layers**: [64, 32, 16, 8] works well (decreasing)
3. **Pre-training**: Pre-train GMF and MLP separately, then combine
4. ** Negative Sampling**: 4-10 negatives per positive
5. **Regularization**: L2 on embeddings (1e-6 to 1e-4)
6. **Activation**: ReLU or LeakyReLU in MLP
7. **Loss**: Binary cross-entropy for implicit feedback
8. **Learning Rate**: 0.001 often sufficient""",
        "dataset": "movielens-1m",
        "init_params": """    embedding_dim=32,
    mlp_layers=[64, 32, 16, 8],
    gmf_dim=32,
    pretrain=False,
    dropout=0.0,""",
    },
    "LightGCN": {
        "full_name": "Light Graph Convolutional Network",
        "paper": "He et al. 2020 - LightGCN: Simplifying and Powering Graph Convolution Network",
        "intro": "simplifies graph convolutional networks by removing feature transformation and non-linear activation, keeping only neighborhood aggregation for collaborative filtering",
        "architecture": """**Simplified GCN for Collaborative Filtering:**

1. **Graph Construction**: User-item bipartite graph
2. **Light Graph Convolution** (L layers):
   - Pure neighborhood aggregation
   - NO feature transformation
   - NO non-linear activation
3. **Layer Combination**: Weighted sum of all layers
4. **Prediction**: Inner product of final embeddings

**Key Innovation**: Shows that feature transformation and activation hurt CF performance

**Architecture:**
```
Graph → Embed → Aggregate(1) → ... → Aggregate(L) → Combine Layers → Predict
                   ↓                                        ↓
           (no transformation)                    (weighted average)
```""",
        "math": """**Light Graph Convolution Layer:**
```
e_u^(k+1) = Σ_{i ∈ N_u} (1/√|N_u||N_i|) · e_i^(k)
e_i^(k+1) = Σ_{u ∈ N_i} (1/√|N_i||N_u|) · e_u^(k)
```

**NO transformation matrix W**
**NO activation function σ**

**Layer Combination:**
```
e_u = Σ_{k=0}^K α_k · e_u^(k)
e_i = Σ_{k=0}^K α_k · e_i^(k)
```
where α_k = 1/(K+1) (uniform weighting)

**Prediction:**
```
ŷ_ui = e_u^T · e_i
```""",
        "use_cases": """✅ **Perfect For:**
- Large-scale collaborative filtering (millions of users/items)
- Implicit feedback data
- When you want GNN benefits with MF simplicity
- Sparse bipartite graphs
- Top-N recommendation at scale

❌ **Avoid When:**
- Have rich node features → use full GCN
- Heterogeneous graphs with multiple edge types
- Need to model side information
- Graph structure is not beneficial""",
        "best_practices": """1. **Number of Layers**: 2-4 layers (3 often optimal)
2. **Embedding Dimension**: 64 standard, 128-256 for large datasets
3. **Layer Combination**: Uniform weights work well (α_k = 1/(K+1))
4. **Negative Sampling**: Sample 1000-2000 negatives per positive (more than NCF!)
5. **Dropout**: Not needed (architecture is simple enough)
6. **Learning Rate**: 0.001 with decay
7. **Batch Size**: Large (2048-8192) for stability
8. **Regularization**: L2=1e-4 on embeddings""",
        "dataset": "gowalla",  # location-based
        "init_params": """    embedding_dim=64,
    num_layers=3,
    reg_weight=1e-4,
    negative_samples=1000,""",
    },
    "DIEN": {
        "full_name": "Deep Interest Evolution Network",
        "paper": "Zhou et al. 2019 - Deep Interest Evolution Network",
        "intro": "models the evolution of user interests over time using AUGRU (Attention-based GRU) to capture interest dynamics",
        "architecture": """**Interest Evolution with Attention:**

1. **Interest Extractor Layer**:
   - GRU over user behavior sequence
   - Extracts interest states at each time step
   
2. **Interest Evolving Layer**:
   - AUGRU (Attention-based GRU) with attention scores
   - Models how interests evolve toward target item
   - Attention controls update gate based on relevance

3. **Final Layer**: Combines evolved interests with other features

**Key Innovation**: Uses attention to model interest evolution, not just representation

**Architecture:**
```
Behavior Seq → GRU (Interest Extract) → AUGRU (Evolution) → Final Interest → Predict
                                          ↑
                                    Attention from Target
```""",
        "math": """**Interest Extractor (GRU):**
```
h_t = GRU(e_t, h_{t-1})
```

**Attention Score:**
```
a_t = softmax(e_target^T · W · h_t)
```

**AUGRU Update:**
```
u_t' = a_t · u_t               # attention-weighted update gate
h_t' = (1 - u_t') ⊙ h_{t-1} + u_t' ⊙ h̃_t
```
where h̃_t is candidate state

**Final Representation:**
```
interest = Σ_t a_t · h_t'
```""",
        "use_cases": """✅ **Ideal For:**
- E-commerce with browsing sequences
- Interest drift is important (fashion, trends)
- Long user sequences (20-100 items)
- When you need to understand WHY user clicked
- Evolving user preferences

❌ **Not For:**
- Static preferences
- Short sequences (<10 items)
- Real-time constraints (AUGRU is slower)
- Simple collaborative filtering tasks""",
        "best_practices": """1. **Sequence Length**: 20-50 items (longer than DIN)
2. **GRU Hidden Dim**: 36-64 typical
3. **Attention Type**: Scaled dot-product works well
4. **Auxiliary Loss**: Add on intermediate interest states
5. **Negative Sampling**: Sample from non-clicked items in session
6. **Feature Fusion**: Combine with user/item features
7. **Learning Rate**: 0.001, decay after epochs
8. **Training**: Expensive - use GPU acceleration""",
        "dataset": "amazon-electronics",
        "init_params": """    gru_hidden_dim=36,
    attention_hidden_dim=36,
    seq_len=50,
    use_auxiliary_loss=True,""",
    },
    "DIN": {
        "full_name": "Deep Interest Network",
        "paper": "Zhou et al. 2018 - Deep Interest Network for Click-Through Rate Prediction",
        "intro": "uses local activation (attention) to adaptively learn user interests from historical behaviors based on the candidate item",
        "architecture": """**Adaptive Interest Activation:**

1. **Behavior Representation**: Embed user's historical items
2. **Local Activation Unit**:
   - Calculates attention weights between each historical item and target item
   - Relevant items get higher weights
3. **Weighted Pooling**: Sum historical embeddings weighted by attention
4. **Final MLP**: Combines activated interest with other features

**Key Innovation**: Not all behaviors are equally relevant - activate based on target item

**Architecture:**
```
Historical Items → Embed → Attention(target) → Weighted Sum → \
                                                              Concat → MLP → Prediction
Target Item → Embed ────────────────────────────────────────→ /

Other Features ──────────────────────────────────────────────→ /
```""",
        "math": """**Attention Weight Calculation:**
```
a(e_i, e_target) = MLP([e_i; e_target; e_i ⊙ e_target; e_i - e_target])
α_i = exp(a_i) / Σ_j exp(a_j)
```

**Activated User Representation:**
```
Interest = Σ_i α_i · e_i
```

**Final Prediction:**
```
ŷ = σ(MLP([Interest; User_features; Item_features; Context]))
```""",
        "use_cases": """✅ **Perfect For:**
- CTR prediction with user history
- Diverse user behaviors (browsing different categories)
- When relevance to target matters
- Short-to-medium sequences (5-30 items)
- Display advertising, e-commerce

❌ **Not Suitable For:**
- No user history available
- All items equally relevant
- Very long sequences (use interest evolution like DIEN)
- Static collaborative filtering""",
        "best_practices": """1. **Sequence Length**: 5-30 items (shorter than DIEN/SASRec)
2. **Embedding Dimension**: 8-32 for items, 4-16 for features
3. **Attention MLP**: [64, 32] typically sufficient
4. **Main MLP**: [256, 128, 64, 32]
5. **Activation**: PReLU (from paper) or Dice activation
6. **Batch Normalization**: Use adaptive BN across mini-batches
7. **Regularization**: Dropout (0.1-0.2) + L2 on embeddings
8. **Negative Sampling**: Important for training efficiency""",
        "dataset": "amazon-ads",
        "init_params": """    item_embed_dim=16,
    attention_mlp=[64, 32],
    deep_mlp=[256, 128, 64, 32],
    activation='dice',
    use_bn=True,""",
    },
    # ========== BATCH 2: NEXT 10 MODELS ==========
    "BPR": {
        "full_name": "Bayesian Personalized Ranking",
        "paper": "Rendle et al. 2009 - BPR: Bayesian Personalized Ranking from Implicit Feedback",
        "intro": "optimizes for pairwise ranking using a Bayesian approach, assuming users prefer observed items over unobserved ones",
        "architecture": """**Pairwise Ranking Optimization:**

1. **Assumption**: User prefers observed item i over unobserved item j
2. **Pairwise Comparisons**: For each user, create positive-negative pairs
3. **Matrix Factorization**: Learn user/item embeddings
4. **Ranking Loss**: Optimize probability that positive > negative

**Key Innovation**: Learns to rank rather than predict ratings

**Architecture:**
```
User u, Item i (pos), Item j (neg) → Embeddings → 
    Score(u,i) - Score(u,j) → Sigmoid → Loss
```""",
        "math": """**Scoring Function:**
```
x̂_uij = x̂_ui - x̂_uj
where x̂_ui = <p_u, q_i>  # dot product
```

**BPR-Opt Criterion:**
```
BPR-OPT = Σ_{(u,i,j)} ln σ(x̂_uij) - λ||Θ||²
```
where σ is sigmoid, Θ are parameters

**Gradient Update:**
```
∂BPR/∂θ = Σ_{(u,i,j)} σ(-x̂_uij) · ∂x̂_uij/∂θ - λθ
```

**Sampling Strategy:**
- For each user u
- Sample observed item i (positive)
- Sample unobserved item j (negative)
- Update to maximize x̂_ui - x̂_uj""",
        "use_cases": """✅ **Perfect For:**
- Implicit feedback (clicks, views, purchases)
- Top-N recommendation rankings
- When you care about order, not exact ratings
- Large item catalogs (need to rank many items)
- Personalized rankings

❌ **Not For:**
- Explicit ratings (use rating prediction models)
- Need probability calibration
- Very sparse data (try simpler MF first)
- Sequential patterns (use SASRec)""",
        "best_practices": """1. **Negative Sampling**: Sample 1-10 negatives per positive
2. **Learning Rate**: 0.01-0.05 typical (higher than supervised)
3. **Regularization**: λ = 0.01-0.001
4. **Sampling Strategy**: Uniform or popularity-based for negatives
5. **Batch Size**: 256-1024 for stability
6. **Convergence**: Monitor pairwise accuracy, not loss
7. **Embedding Dim**: 20-100 usually sufficient
8. **Update Frequency**: Bootstrap sampling each epoch""",
        "dataset": "movielens-1m",
        "init_params": """    embedding_dim=50,
    learning_rate=0.05,
    reg=0.01,
    num_negatives=5,""",
    },
    "SVD": {
        "full_name": "Singular Value Decomposition",
        "paper": "Simon Funk 2006 - Netflix Prize; Koren 2008 - Factorization Meets Neighborhood",
        "intro": "decomposes the user-item rating matrix into user and item latent factor matrices using gradient descent optimization",
        "architecture": """**Classic Matrix Factorization:**

1. **Decomposition**: R ≈ U × V^T
   - U: user latent factors (m × k)
   - V: item latent factors (n × k)
2. **Prediction**: Rating = dot(user_factors, item_factors) + biases
3. **Optimization**: Minimize squared error with regularization

**Key Innovation**: Efficient factorization for sparse matrices

**Architecture:**
```
User u → [user_vector_u] \
                          dot product + biases → Prediction
Item i → [item_vector_i] /
```""",
        "math": """**Prediction Formula:**
```
r̂_ui = μ + b_u + b_i + q_i^T · p_u
```
where:
- μ: global mean rating
- b_u: user bias
- b_i: item bias
- p_u: user latent factors (k-dim)
- q_i: item latent factors (k-dim)

**Loss Function:**
```
L = Σ_{(u,i)∈R} (r_ui - r̂_ui)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
```

**Gradient Updates:**
```
e_ui = r_ui - r̂_ui
p_u ← p_u + α(e_ui · q_i - λ · p_u)
q_i ← q_i + α(e_ui · p_u - λ · q_i)
b_u ← b_u + α(e_ui - λ · b_u)
b_i ← b_i + α(e_ui - λ · b_i)
```""",
        "use_cases": """✅ **Excellent For:**
- Explicit ratings (1-5 stars)
- Baseline collaborative filtering
- Medium-scale datasets (10K-10M ratings)
- Rating prediction tasks
- Well-understood, interpretable results

❌ **Avoid When:**
- Implicit feedback (use BPR instead)
- Need deep interactions (use neural models)
- Sequential patterns (use RNNs)
- Very large scale (>100M ratings) - use ALS instead""",
        "best_practices": """1. **Factors (k)**: Start with 20-100, more for complex patterns
2. **Learning Rate**: 0.005-0.01 typical
3. **Regularization**: 0.02-0.1 to prevent overfitting
4. **Initialization**: Random normal (0, 0.1)
5. **Biases**: Always include global and user/item biases!
6. **Convergence**: Monitor RMSE on validation set
7. **Early Stopping**:Stop when validation RMSE increases
8. **Epochs**: 20-50 usually sufficient""",
        "dataset": "movielens-100k",
        "init_params": """    n_factors=100,
    learning_rate=0.005,
    reg=0.02,
    n_epochs=30,""",
    },
    "RBM": {
        "full_name": "Restricted Boltzmann Machine",
        "paper": "Salakhutdinov et al. 2007 - Restricted Boltzmann Machines for Collaborative Filtering",
        "intro": "uses a probabilistic graphical model with visible (ratings) and hidden (latent features) units to learn user preferences",
        "architecture": """**Energy-Based Probabilistic Model:**

1. **Visible Units**: User ratings for items
2. **Hidden Units**: Latent user preferences
3. **Bipartite Structure**: No visible-visible or hidden-hidden connections
4. **Training**: Contrastive Divergence (CD-k)

**Key Innovation**: Generative model that can sample new ratings

**Architecture:**
```
Visible (Ratings) ↔ Weights W ↔ Hidden (Features)
      ↓                              ↓
  Biases a                      Biases b
```

No connections within visible or hidden layers!""",
        "math": """**Energy Function:**
```
E(v,h) = -Σ_i a_i·v_i - Σ_j b_j·h_j - Σ_i Σ_j v_i·h_j·w_ij
```

**Probability:**
```
P(v,h) = exp(-E(v,h)) / Z
where Z = Σ_{v',h'} exp(-E(v',h'))  # partition function
```

**Conditional Probabilities:**
```
P(h_j=1|v) = σ(b_j + Σ_i v_i·w_ij)
P(v_i=1|h) = σ(a_i + Σ_j h_j·w_ij)
```

**Contrastive Divergence Update:**
```
ΔW = ε(<v_0 h_0^T> - <v_k h_k^T>)
```
where v_0 is data, v_k is reconstruction after k Gibbs steps""",
        "use_cases": """✅ **Ideal For:**
- Explicit ratings (discrete values)
- Need generative capability (sample ratings)
- Feature learning from ratings
- Cold-start scenarios (can infer from partial data)
- Research and experimentation

❌ **Not For:**
- Large-scale production (slow training)
- Implicit feedback
- need online updates (batch model)
- Interpretability required""",
        "best_practices": """1. **Hidden Units**: 100-500 typical
2. **CD Steps (k)**: k=1 works well (CD-1)
3. **Learning Rate**: 0.01-0.1, use learning rate decay
4. **Momentum**: 0.5 → 0.9 over training
5. **Weight Decay**: 0.0001-0.001
6. **Mini-batch**: 100-1000 users per batch
7. **Epochs**: 50-200 epochs
8. **Initialization**: Small random weights ~N(0, 0.01)""",
        "dataset": "movielens-1m",
        "init_params": """    n_hidden=200,
    learning_rate=0.01,
    momentum=0.5,
    n_epochs=100,
    cd_steps=1,""",
    },
    # ========== BATCH 2 REMAINING ==========
    "AutoInt": {
        "full_name": "Automatic Feature Interaction Learning via Self-Attention",
        "paper": "Song et al. 2019 - AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks",
        "intro": "uses multi-head self-attention to automatically learn high-order feature interactions without manual feature engineering",
        "architecture": """**Self-Attention for Feature Interactions:**

1. **Embedding Layer**: Convert sparse features to dense embeddings
2. **Interacting Layer**: Multi-head self-attention on feature embeddings
   - Each head learns different interaction patterns
   - Residual connections preserve original features
3. **Stacking**: Multiple attention layers for higher-order interactions
4. **Output Layer**: Combine all layers for prediction

**Key Innovation**: Treats features as a sequence, applies self-attention

**Architecture:**
```
Features → Embed → [Self-Attention + Residual]×L → Concat → Output
```""",
        "math": """**Multi-Head Self-Attention:**
```
head_h = Attention(E·W_h^Q, E·W_h^K, E·W_h^V)
where E = [e_1, e_2, ..., e_m]  # feature embeddings
```

**Attention Mechanism:**
```
Attention(Q,K,V) = softmax((Q·K^T)/√d) · V
```

**Multi-Head Output:**
```
MultiHead(E) = [head_1; head_2; ...; head_H] × W^O
```

**Residual Connection:**
```
E' = ReLU(MultiHead(E) + E)
```

**Final Prediction:**
```
ŷ = σ(w^T · [E^(0); E^(1); ...; E^(L)])
```
where E^(l) is output of layer l""",
        "use_cases": """✅ **Excellent For:**
- Feature-rich CTR prediction
- When manual feature engineering is expensive
- Discovering unknown feature interactions
- Sparse categorical features
- Display advertising, app recommendations

❌ **Not For:**
- Simple datasets (overkill)
- No features available (use CF)
- Sequential patterns (use RNN)
- Very large feature spaces (memory intensive)""",
        "best_practices": """1. **Attention Heads**: 2-4 heads sufficient
2. **Layers**: 2-3 interacting layers
3. **Embedding Dim**: 16-64 per feature
4. **Attention Dim**: Same as embedding
5. **Residual**: Always use residual connections
6. **Dropout**: 0.1-0.2 on attention weights
7. **Learning Rate**: 0.001 with warmup
8. **Batch Size**: 256-1024""",
        "dataset": "crite-mini",
        "init_params": """    embedding_dim=32,
    attention_dim=32,
    num_heads=2,
    num_layers=3,
    dropout=0.1,""",
    },
    "ALS": {
        "full_name": "Alternating Least Squares",
        "paper": "Hu et al. 2008 - Collaborative Filtering for Implicit Feedback",
        "intro": "alternates between fixing user factors and solving for item factors (and vice versa) to efficiently factorize large sparse matrices",
        "architecture": """**Alternating Optimization for Scalability:**

1. **Initialization**: Random user and item factors
2. **Fix Items, Solve Users**: Solve for all user factors in parallel
3. **Fix Users, Solve Items**: Solve for all item factors in parallel
4. **Repeat**: Alternate until convergence

**Key Innovation**: Closed-form solution per iteration, highly parallelizable

**Objective:**
```
min Σ_(u,i) c_ui(p_ui - x_u^T y_i)² + λ(Σ||x_u||² + Σ||y_i||²)
```""",
        "math": """**User Update (parallel over all users):**
```
x_u = (Y^T C^u Y + λI)^(-1) Y^T C^u p^u
```
where:
- Y: item factor matrix
- C^u: confidence diagonal matrix for user u
- p^u: preference vector for user u

**Item Update (parallel over all items):**
```
y_i = (X^T C^i X + λI)^(-1) X^T C^i p^i
```

**Confidence:**
```
c_ui = 1 + α · r_ui
```
for implicit feedback r_ui""",
        "use_cases": """✅ **Perfect For:**
- Very large scale (billions of interactions)
- Implicit feedback (views, clicks)
- Distributed/parallel computation
- Production systems at scale
- Spark, Hadoop environments

❌ **Not For:**
- Small datasets (<10K) - SGD simpler
- Explicit ratings - SGD more flexible
- Need online updates - ALS is batch
- Dense features - use neural models""",
        "best_practices": """1. **Factors**: 50-200 typically
2. **Confidence α**: 40 is common (tune 10-100)
3. **Regularization λ**: 0.01-0.1
4. **Iterations**: 10-20 sufficient
5. **Parallelization**: Partition by user/item blocks
6. **Implicit**: Always use confidence weighting
7. **Caching**: Cache Y^T Y and X^T X
8. **Convergence**: Monitor RMSE on validation""",
        "dataset": "spotify-1m",
        "init_params": """    n_factors=100,
    regularization=0.01,
    alpha=40,
    iterations=15,""",
    },
    "Bert4Rec": {
        "full_name": "BERT for Sequential Recommendation",
        "paper": "Sun et al. 2019 - BERT4Rec: Sequential Recommendation with Bidirectional Encoder",
        "intro": "applies bidirectional transformer (BERT) to user sequences using cloze task to learn from both past and future context",
        "architecture": """**Bidirectional Transformer:**

1. **Masked Sequence Modeling**: Randomly mask 15-20% items
2. **Bidirectional Attention**: Full attention (no causal mask)
3. **Transformer Encoder**: Stacked self-attention + FFN
4. **Prediction Head**: Predict masked items

**Key vs SASRec**: Can look both directions (not just left-to-right)

**Architecture:**
```
[i1, [MASK], i3, i4, [MASK]] → Transformer → Predict [i2, i5]
```""",
        "math": """**Full Bidirectional Attention:**
```
Attention(Q,K,V) = softmax(QK^T/√d) · V
```
No causal masking!

**Cloze Training Objective:**
```
L = -Σ_{m∈masked} log P(v_m | S_{\\m})
```
where S_{\\m} is sequence excluding position m

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos/10000^(2i/d))
PE(pos, 2i+1) = cos(pos/10000^(2i/d))
```""",
        "use_cases": """✅ **Perfect For:**
- Offline training with full sequences
- Long interaction histories (50-200 items)
- Rich bidirectional patterns
- Cold start with partial sequences
- Research and benchmarking

❌ **Not For:**
- Real-time next-item prediction (needs future context)
- Online/streaming scenarios
- Very short sequences (<10 items)
- Production serving (slow inference)""",
        "best_practices": """1. **Mask Probability**: 15-20% of items
2. **Max Sequence Length**: 50-200 items
3. **Transformer Layers**: 2-4 layers
4. **Attention Heads**: 2-4 heads
5. **Hidden Dimension**: 64-128
6. **Warmup Steps**: 1000-10000 for learning rate
7. **Batch Size**: 128-512
8. **Pre-training**: Can pre-train on large corpus""",
        "dataset": "amazon-books",
        "init_params": """    max_seq_len=50,
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    mask_prob=0.15,""",
    },
    "DLRM": {
        "full_name": "Deep Learning Recommendation Model",
        "paper": "Naumov et al. 2019 - Deep Learning Recommendation Model (Facebook)",
        "intro": "separates dense and sparse feature processing, using explicit pairwise dot product interactions between embeddings",
        "architecture": """**Parallel Dense/Sparse Processing:**

1. **Bottom MLP**: Process continuous features
2. **Embedding Layers**: Parallel lookup for categorical features
3. **Explicit Interactions**: Dot products between all embedding pairs
4. **Top MLP**: Process concatenated features + interactions

**Key Innovation**: Scalable architecture for production (Facebook, Pinterest)

**Architecture:**
```
Dense → Bottom MLP →  \
                      [Dot Products] → Concat → Top MLP → Pred
Sparse → Embed×K → /
```""",
        "math": """**Dense Processing:**
```
z_dense = MLP_bottom(x_dense)
```

**Embedding Interactions** (all pairs):
```
I = {<e_i, e_j> : for all i < j}
where <·,·> is dot product
```

**Concatenation:**
```
z = [z_dense; e_1; e_2; ...; e_K; I]
```

**Final Prediction:**
```
ŷ = σ(MLP_top(z))
```""",
        "use_cases": """✅ **Ideal For:**
- Large-scale CTR prediction (billions of users)
- Production systems (Facebook, Pinterest scale)
- Mix of dense & sparse features
- Parallelizable infrastructure
- High-throughput serving

❌ **Not For:**
- Small datasets (<100K)
- Pure collaborative filtering
- Sequential patterns
- Limited compute/memory""",
        "best_practices": """1. **Bottom MLP**: [512, 256, 64]
2. **Top MLP**: [512, 512, 256, 1]
3. **Embedding Dim**: 16-128 (by cardinality)
4. **Batch Size**: 2048-8192 (very large!)
5. **Mixed Precision**: Use FP16 for speed
6. **Parallelization**: Embeddings in parallel
7. **Caching**: Cache popular item embeddings
8. **Hardware**: Multi-GPU or TPU""",
        "dataset": "criteo",
        "init_params": """    bottom_mlp=[512, 256, 64],
    top_mlp=[512, 512, 256, 1],
    embedding_dim=64,""",
    },
    "FFM": {
        "full_name": "Field-aware Factorization Machine",
        "paper": "Juan et al. 2016 - Field-aware Factorization Machines for CTR Prediction",
        "intro": "extends FM with field-aware latent vectors where each feature has different embeddings for interacting with different fields",
        "architecture": """**Field-Aware Interactions:**

1. **Feature Fields**: Group features (e.g., User, Item, Context)
2. **Field-Specific Embeddings**: v_{i,f_j} for feature i w.r.t. field j
3. **Pairwise Interactions**: Use appropriate field-specific vectors

**Key Innovation**: Different latent factors per field pair

**Example:**
- Publisher × Advertiser uses v_{pub,adv} and v_{adv,pub}
- Publisher × Gender uses v_{pub,gender} and v_{gender,pub}""",
        "math": """**FFM Formula:**
```
ŷ = w_0 + Σ_i w_i·x_i + Σ_i Σ_{j>i} <v_{i,f_j}, v_{j,f_i}>·x_i·x_j
```
where:
- v_{i,f_j}: latent vector for feature i w.r.t. field of feature j
- f_j: field of feature j

**vs Standard FM:**
```
FM:  <v_i, v_j>  # same vector always
FFM: <v_{i,f_j}, v_{j,f_i}>  # field-aware
```""",
        "use_cases": """✅ **Perfect For:**
- CTR prediction with field structure
- Sparse categorical features
- When feature fields matter (User/Item/Context)
- Kaggle competitions
- Display advertising

❌ **Not For:**
- No natural field structure
- Dense numerical features
- Need deep interactions
- Very large scale (>10M features)""",
        "best_practices": """1. **Latent Factors**: k=4-8 (smaller than FM!)
2. **Optimizer**: AdaGrad or FTRL
3. **Learning Rate**: 0.1-0.2 (higher than FM)
4. **Regularization**: λ = 0.00002 typical
5. **Early Stopping**: Monitor validation AUC
6. **Normalization**: Normalize continuous features
7. **Field Definition**: Carefully design fields
8. **Memory**: k×fields larger than FM""",
        "dataset": "criteo",
        "init_params": """    n_factors=4,
    learning_rate=0.2,
    reg=0.00002,""",
    },
    "Caser": {
        "full_name": "Convolutional Sequence Embedding Recommendation",
        "paper": "Tang & Wang 2018 - Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding",
        "intro": 'treats user sequence as an "image" applying CNNs with horizontal (recent patterns) and vertical (skip patterns) convolutional filters',
        "architecture": """**CNN for Sequential Patterns:**

1. **Sequence Embedding Matrix**: L×d (sequence × embedding)
2. **Horizontal Convolution**: Capture recent sequential patterns
3. **Vertical Convolution**: Capture skip patterns (point-level)
4. **Pooling**: Max pooling over all filters
5. **Fully Connected**: Combine features for prediction

**Key Innovation**: CNN treats sequence as "image"

**Architecture:**
```
Sequence Matrix → [Horiz Conv] → Pool → \
                                         Concat → FC → Predict
                  [Vert Conv]  → Pool → /
```""",
        "math": """**Horizontal Convolution** (height h):
```
c^h = ReLU(W^h * E_{i:i+h} + b^h)
```
Captures last h items together

**Vertical Convolution** (width 1):
```
c^v = ReLU(W^v * E_i + b^v)
```
Captures single-point patterns

**Output:**
```
z = [max(c^h₁); ...; max(c^hₙ); max(c^v₁); ...; max(c^vₘ)]
y = FC(z)
```""",
        "use_cases": """✅ **Ideal For:**
- Short-medium sequences (5-50 items)
- Local sequential patterns
- Recent behavior matters most
- E-commerce browsing sessions
- When RNN/Transformer is overkill

❌ **Not For:**
- Very long sequences (>100 items)
- Global long-range dependencies
- When position encoding crucial
- Sparse interaction data""",
        "best_practices": """1. **Sequence Length**: L = 5-20 recent items
2. **Horizontal Filters**: heights = [2, 3, 4, 5]
3. **Vertical Filters**: 4-16 filters
4. **Filter Numbers**: 16-64 per size
5. **Dropout**: 0.5 on FC layer
6. **Embedding Dim**: 50-100
7. **Activation**: ReLU standard
8. **Pooling**: Max pooling works best""",
        "dataset": "amazon-movies",
        "init_params": """    seq_len=10,
    horiz_filters=[2, 3, 4],
    num_filters=16,
    dropout=0.5,""",
    },
    "BiVAE": {
        "full_name": "Bilateral Variational Autoencoder",
        "paper": "Collaborative Filtering via Variational Autoencoders",
        "intro": "uses variational inference to jointly learn latent representations for both users and items with uncertainty quantification",
        "architecture": """**Dual Variational Autoencoder:**

1. **User Encoder**: q(z_u | r_u) → μ_u, σ_u
2. **Item Encoder**: q(z_i | r_i) → μ_i, σ_i
3. **Sampling**: z_u ~ N(μ_u, σ_u²), z_i ~ N(μ_i, σ_i²)
4. **Decoder**: p(r | z_u, z_i) reconstructs rating
5. **KL Regularization**: Towards prior N(0,I)

**Key Innovation**: Probabilistic latent factors with uncertainty

**Architecture:**
```
User → Encode → μ_u, σ_u → Sample → z_u \
                                          Decode → Rating
Item → Encode → μ_i, σ_i → Sample → z_i /
```""",
        "math": """**ELBO (Evidence Lower Bound):**
```
L = E_{z~q}[log p(r|z_u,z_i)] - KL(q(z_u,z_i) || p(z_u)p(z_i))
```

**Reconstruction:**
```
log p(r|z_u,z_i) = -||r - f(z_u,z_i)||²
```

**KL Divergence:**
```
KL = -0.5 Σ(1 + log σ² - μ² - σ²)
```

**Reparameterization Trick:**
```
z = μ + σ ⊙ ε,  where ε ~ N(0,I)
```""",
        "use_cases": """✅ **Good For:**
- Uncertainty quantification
- Generative modeling (sample ratings)
- Cold-start with priors
- Small-medium datasets
- Research and experimentation

❌ **Not For:**
- Large-scale production (slow)
- Need interpretability
- Real-time inference
- When point estimates suffice""",
        "best_practices": """1. **Latent Dimension**: 20-50
2. **KL Annealing**: Start with β=0.01 → 1.0
3. **Encoder**: [200, 100] typical
4. **Decoder**: Symmetric [100, 200]
5. **Activation**: Tanh or ReLU
6. **Learning Rate**: 0.001 with decay
7. **Warmup**: 10-20 epochs before full KL
8. **Batch Size**: 128-512""",
        "dataset": "movielens-100k",
        "init_params": """    latent_dim=20,
    encoder_dims=[200, 100],
    kl_anneal_rate=0.01,""",
    },
    "AFM": {
        "full_name": "Attentional Factorization Machine",
        "paper": "Xiao et al. 2017 - Attentional Factorization Machines",
        "intro": "adds attention mechanism to FM to learn the importance of different feature interactions automatically",
        "architecture": """**FM + Attention:**

1. **Feature Embeddings**: v_i for each feature
2. **Element-wise Products**: v_i ⊙ v_j for all pairs
3. **Attention Network**: Learn weights for each interaction
4. **Weighted Sum**: Combine interactions by importance

**Key Innovation**: Not all feature interactions equally important

**Architecture:**
```
Features → Embed → [Pairwise Products] → Attention → Sum → Linear → Pred
```""",
        "math": """**Attention-Based Pooling:**
```
α_ij = h^T ReLU(W(v_i ⊙ v_j) + b)
a_ij = exp(α_ij) / Σ exp(α_kl)
```

**AFM Formula:**
```
ŷ = w_0 + Σ w_i·x_i + p^T Σ_{i,j} a_ij(v_i ⊙ v_j)x_i x_j
```
where:
- ⊙ is element-wise product
- a_ij is attention weight for (i,j) interaction
- p is projection vector

**vs FM:**
```
FM:  Σ <v_i, v_j>  # equal weight
AFM: Σ a_ij(v_i ⊙ v_j)  # learned weight
```""",
        "use_cases": """✅ **Perfect For:**
- Sparse feature interactions
- CTR prediction
- When interactions vary in importance
- Interpretable recommendations
- Feature engineering insights

❌ **Not For:**
- Need very deep interactions (use DeepFM)
- Sequential patterns (use RNN)
- Dense feature spaces
- Very large scale (slower than FM)""",
        "best_practices": """1. **Embedding Dim**: 8-16 (smaller than FM)
2. **Attention Dim**: 32-64
3. **Dropout**: 0.5-0.7 on attention layer
4. **Regularization**: L2 = 1e-6 to 1e-4
5. **Learning Rate**: 0.001-0.01
6. **Batch Normalization**: After attention
7. **Activation**: ReLU in attention network
8. **Visualization**: Analyze learned attention weights""",
        "dataset": "frappe",
        "init_params": """    embedding_dim=10,
    attention_dim=64,
    dropout=0.6,
    reg=1e-6,""",
    },
    # ========== BATCH 3 FINAL ==========
    "A2SVD": {
        "full_name": "Adaptive Singular Value Decomposition",
        "paper": "Koren 2008 - Factorization Meets Neighborhood",
        "intro": "extends SVD with adaptive regularization and neighborhood-based refinements for better generalization",
        "architecture": """**Enhanced SVD with Adaptivity:**

1. **Base SVD**: Standard latent factor decomposition
2. **Adaptive Regularization**: Per-user/item regularization
3. **Neighborhood Integration**: Add implicit feedback signals
4. **Bias Modeling**: Global, user, item, and temporal biases

**Key Innovation**: Adaptive λ based on user/item activity

**Architecture:**
```
SVD Factors + Neighborhood Implicit + Temporal Bias → Prediction
```""",
        "math": """**Adaptive Prediction:**
```
r̂_ui = μ + b_u(t) + b_i(t) + q_i^T(p_u + |N(u)|^(-0.5) Σ_{j∈N(u)} y_j)
```
where:
- μ: global mean
- b_u(t), b_i(t): time-dependent biases
- p_u: user factors
- q_i: item factors
- y_j: implicit factors from neighbors

**Adaptive Regularization:**
```
λ_u = λ_0 + β/√|R_u|
λ_i = λ_0 + β/√|R_i|
```
Lower regularization for active users/items""",
        "use_cases": """✅ **Excellent For:**
- Explicit ratings with temporal dynamics
- Users/items with varying activity levels
- When simple SVD underfits active users
- Netflix-style datasets
- Combining explicit + implicit signals

❌ **Not For:**
- Pure implicit feedback (use BPR)
- Need real-time updates
- Very small datasets (regularization complex)
- When simple SVD works fine""",
        "best_practices": """1. **Base Factors**: k=100-200
2. **Adaptive λ**: β=0.1-1.0
3. **Neighborhood Size**: 50-100 similar items
4. **Temporal Bins**: Week or month-level
5. **Learning Rate**: 0.001-0.01
6. **Epochs**: 30-50
7. **Combine Signals**: Weight implicit feedback 0.1-0.3
8. **Bias Priority**: Optimize biases first""",
        "dataset": "movielens-1m",
        "init_params": """    n_factors=150,
    base_reg=0.02,
    adaptive_beta=0.5,
    n_neighbors=50,""",
    },
    "BST": {
        "full_name": "Behavior Sequence Transformer",
        "paper": "Chen et al. 2019 - Behavior Sequence Transformer for E-commerce Recommendation",
        "intro": "applies transformer architecture to user behavior sequences with target item attention for personalized recommendation",
        "architecture": """**Transformer for E-commerce:**

1. **Behavior Sequence**: User's historical actions
2. **Multi-Head Self-Attention**: Capture item dependencies
3. **Target Attention**: Attend based on candidate item
4. **Position Encoding**: Capture sequence order
5. **Embedding Concat**: Combine sequence + features

**Key Innovation**: Target-aware attention (like DIN but with Transformer)

**Architecture:**
```
[Seq] → Transformer → Target Attn → [Other Features] → MLP → Pred
                          ↑
                     Candidate Item
```""",
        "math": """**Self-Attention:**
```
H = MultiHead(V_seq)
```

**Target Attention:**
```
α_i = softmax(h_i^T W_q e_target / √d)
v_user = Σ α_i · h_i
```

**Final Prediction:**
```
features = [v_user; e_target; context]
ŷ = MLP(features)
```

**Positional Encoding:**
```
PE(pos) = LearnableEmbed(pos)
```
Learned, not sinusoidal""",
        "use_cases": """✅ **Perfect For:**
- E-commerce with rich sequences
- CTR prediction with behavior history
- Target-aware recommendations
- Medium-length sequences (10-50 items)
- When you need attention visualization

❌ **Not For:**
- Very short sequences (<5 items)
- Pure collaborative filtering
- Real-time constraints (slower than GRU)
- When SASRec already works""",
        "best_practices": """1. **Sequence Length**: 10-50 recent items
2. **Transformer Layers**: 1-2 layers (shallow!)
3. **Attention Heads**: 1-2 heads
4. **Hidden Dim**: 64-128
5. **Target Attention**: Essential for performance
6. **Dropout**: 0.1-0.3
7. **Position Encoding**: Learnable works better
8. **Features**: Combine with user/item features""",
        "dataset": "taobao-clicks",
        "init_params": """    seq_len=20,
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    use_target_attention=True,""",
    },
    "BPRMF": {
        "full_name": "Bayesian Personalized Ranking Matrix Factorization",
        "paper": "Rendle et al. 2009 - BPR-MF",
        "intro": "combines BPR ranking optimization with matrix factorization for implicit feedback collaborative filtering",
        "architecture": """**Ranking-Optimized MF:**

1. **Matrix Factorization**: User × Item latent factors
2. **Pairwise Ranking**: BPR loss on (user, pos_item, neg_item) triplets
3. **Sampling**: Bootstrap sampling for efficiency
4. **Optimization**: SGD with ranking gradients

**Key Innovation**: Standard MF with BPR loss instead of pointwise loss

**Architecture:**
```
[U×I Factors] → <p_u, q_i> - <p_u, q_j> → BPR Loss
where i = positive, j = negative
```""",
        "math": """**Matrix Factorization:**
```
x̂_ui = <p_u, q_i> = Σ_f p_uf · q_if
```

**BPR-MF Loss:**
```
L = -Σ_{(u,i,j)} log σ(x̂_ui - x̂_uj) + λ||Θ||²
```

**Gradients:**
```
∂L/∂p_u = σ(-x̂_uij)(q_j - q_i) + λ·p_u
∂L/∂q_i = σ(-x̂_uij)·p_u + λ·q_i
∂L/∂q_j = σ(-x̂_uij)·(-p_u) + λ·q_j
```""",
        "use_cases": """✅ **Perfect For:**
- Implicit feedback ranking
- Top-N recommendation
- When you want simple MF with ranking
- Cold-start items (better than pointwise)
- Medium-scale CF (100K-10M interactions)

❌ **Not For:**
- Explicit ratings (use SVD)
- Need features (use neural models)
- Very sparse data
- Sequential patterns (use RNN)""",
        "best_practices": """1. **Factors**: 20-100 dimensions
2. **Learning Rate**: 0.05-0.1 (higher than SVD)
3. **Regularization**: 0.01-0.001
4. **Negative Samples**: 1-5 per positive
5. **Sampling**: Uniform or popularity-based
6. **Update Frequency**: Shuffle every epoch
7. **Initialization**: Small random ~N(0, 0.01)
8. **Convergence**: Monitor AUC, not loss""",
        "dataset": "lastfm",
        "init_params": """    n_factors=64,
    learning_rate=0.05,
    reg=0.01,
    num_negatives=3,""",
    },
    "GeoIMC": {
        "full_name": "Geographic Inductive Matrix Completion",
        "paper": "Spatial Matrix Completion",
        "intro": "leverages geographic information and graph structure for location-based recommendations using inductive matrix completion",
        "architecture": """**Spatial-Aware Matrix Completion:**

1. **Geographic Graph**: Locations connected by proximity
2. **Graph Convolution**: Propagate location features
3. **User Preferences**: Latent factors
4. **Inductive Completion**: Generalize to new locations
5. **Spatial Regularization**: Nearby locations similar

**Key Innovation**: Combines CF with geographic proximity

**Architecture:**
```
Location Graph → GCN → Location Embeddings \
                                            → MF → Rating Prediction
User Features  → MLP → User Embeddings     /
```""",
        "math": """**Graph Convolution on Locations:**
```
h_l^(k+1) = σ(Σ_{l'∈N(l)} W^k · h_l'^(k) / |N(l)|)
```

**Matrix Factorization:**
```
r̂_ul = p_u^T · q_l + b_u + b_l
```

**Spatial Regularization:**
```
L_spatial = λ_s · Σ_{l,l'∈neighbors} ||q_l - q_l'||²
```

**Total Loss:**
```
L = L_MF + L_spatial + L_graph
```""",
        "use_cases": """✅ **Perfect For:**
- Point-of-Interest (POI) recommendation
- Location-based services (Yelp, Foursquare)
- Check-in data
- When geography matters
- Cold-start locations (inductive)

❌ **Not For:**
- No geographic structure
- Pure collaborative filtering
- When location doesn't matter (movies, books)
- Very sparse check-ins (<100 per user)""",
        "best_practices": """1. **Graph Construction**: k-NN or radius-based (1-5km)
2. **GCN Layers**: 2-3 layers
3. **MF Factors**: 50-100
4. **Spatial Regularization**: λ_s = 0.1-1.0
5. **Distance Weighting**: Inverse distance or Gaussian
6. **Features**: Include category, popularity
7. **Negative Sampling**: Geographic-aware sampling
8. **Train/Test**: Split by time, not random""",
        "dataset": "gowalla-checkins",
        "init_params": """    n_factors=64,
    gcn_layers=2,
    spatial_reg=0.5,
    neighbor_radius_km=2.0,""",
    },
    # ========== BATCH 4: DEEP LEARNING CLASSICS ==========
    "WideDeep": {
        "full_name": "Wide & Deep Learning",
        "paper": "Cheng et al. 2016 - Wide & Deep Learning for Recommender Systems (Google)",
        "intro": "jointly trains wide linear models for memorization and deep neural networks for generalization",
        "architecture": """**Hybrid Architecture:**

1. **Wide Component**: Generalized Linear Model (GLM)
   - Memorizes co-occurrence of features
   - Uses cross-product transformations
2. **Deep Component**: Feed-Forward Neural Network
   - Generalizes to unseen feature combinations
   - Uses low-dimensional embeddings
3. **Joint Training**: Weighted sum of both components

**Key Innovation**: Combining benefits of memorization (Wide) and generalization (Deep)

**Architecture:**
```
Wide Part (Linear)      Deep Part (DNN)
      ↓                       ↓
    Output <----- Sigmoid(Sum)
```""",
        "math": """**Prediction:**
```
P(y=1|x) = σ(w_wide^T [x, φ(x)] + w_deep^T a^(lf) + b)
```
where:
- φ(x): Cross-product transformations (Wide)
- a^(lf): Output of last deep layer
- σ: Sigmoid function

**Joint Optimization:**
Backpropagate gradients to both parts simultaneously using FTRL (Wide) and AdaGrad (Deep).""",
        "use_cases": """✅ **Perfect For:**
- App stores (Google Play)
- Recommending new & old items
- Large-scale regression/classification
- Mixed feature types (sparse + dense)
- Production systems needing low latency

❌ **Not For:**
- Pure collaborative filtering (no features)
- Sequential patterns (use RNN/Transformer)
- Small datasets (overfitting risk)""",
        "best_practices": """1. **Wide Features**: Cross-product of important categorical features
2. **Deep Features**: Embeddings for sparse, raw values for dense
3. **Optimizers**: FTRL for Wide (sparsity), Adam/AdaGrad for Deep
4. **Hidden Layers**: [1024, 512, 256] typical
5. **Embedding Dim**: 32-128
6. **Batch Size**: Large (thousands)""",
        "dataset": "google-play-apps",
        "init_params": """    wide_features=['genre', 'os'],
    deep_features=['age', 'install_history'],
    hidden_units=[1024, 512, 256],""",
    },
    "YouTubeDNN": {
        "full_name": "YouTube Deep Neural Network",
        "paper": "Covington et al. 2016 - Deep Neural Networks for YouTube Recommendations",
        "intro": "uses a two-stage architecture: candidate generation (retrieval) followed by a deep ranking model",
        "architecture": """**Two-Stage Pipeline:**

1. **Candidate Generation (Retrieval)**:
   - Input: User history, search tokens
   - Output: Hundreds of candidates
   - Model: Extreme Multiclass Classification (Softmax)
   - Approximated by Nearest Neighbor Search in embedding space

2. **Ranking**:
   - Input: Candidates + rich features
   - Output: Precise score (watch time)
   - Model: Deep MLP with calibrated output

**Key Innovation**: Scalable deep learning for billions of videos

**Architecture:**
```
User History -> Average Embed -> MLP -> Softmax (Classify Video ID)
```""",
        "math": """**Candidate Generation (Softmax):**
```
P(w_t = i | U, C) = exp(v_i u) / Σ_j exp(v_j u)
```
where u is user embedding (from MLP), v_i is video embedding.

**Ranking (Weighted Logistic):**
Predict expected watch time using weighted logistic regression.
```
E[T] ≈ exp(w^T x)
```""",
        "use_cases": """✅ **Perfect For:**
- Massive scale (millions/billions of items)
- Video recommendation
- Two-stage systems (Retrieval -> Ranking)
- Implicit feedback (watch history)
- Handling fresh content (age feature)

❌ **Not For:**
- Small datasets
- Explicit ratings
- Simple ranking tasks""",
        "best_practices": """1. **Example Age**: Crucial feature for freshness
2. **Negative Sampling**: Importance sampling for Softmax
3. **Embeddings**: Average user's watch history embeddings
4. **Hidden Layers**: ReLU "tower" structure [1024, 512, 256]
5. **Input Features**: Normalize continuous features""",
        "dataset": "youtube-8m",
        "init_params": """    embedding_dim=64,
    hidden_units=[1024, 512, 256],
    n_classes=10000,""",
    },
    "NFM": {
        "full_name": "Neural Factorization Machine",
        "paper": "He et al. 2017 - Neural Factorization Machines for Sparse Predictive Analytics",
        "intro": "seamlessly combines the linearity of FM with the non-linearity of neural networks using a Bi-Interaction pooling layer",
        "architecture": """**Bi-Interaction Layer:**

1. **Embedding**: Sparse features to dense vectors
2. **Bi-Interaction**: Element-wise product of embedding pairs (pooling)
3. **Hidden Layers**: MLP to learn high-order non-linear interactions
4. **Prediction**: Linear part + Neural part

**Key Innovation**: Bi-Interaction layer captures 2nd-order interactions before deep layers

**Architecture:**
```
Embeddings -> Bi-Interaction Pooling -> MLP -> Prediction
```""",
        "math": """**Bi-Interaction Pooling:**
```
f_BI(V_x) = Σ_i Σ_{j>i} x_i v_i ⊙ x_j v_j
          = 0.5 * [(Σ x_i v_i)^2 - Σ (x_i v_i)^2]
```
Linear time complexity O(k)!

**Model Prediction:**
```
ŷ = w_0 + Σ w_i x_i + h^T σ_L(...σ_1(f_BI(V_x))...)
```""",
        "use_cases": """✅ **Perfect For:**
- Sparse predictive analytics
- CTR prediction
- Capturing high-order feature interactions
- When FM is not expressive enough
- General replacement for FM/DeepFM

❌ **Not For:**
- Dense numerical data only
- Sequential data
- Image/Audio inputs""",
        "best_practices": """1. **Dropout**: Essential after Bi-Interaction layer
2. **Batch Norm**: Helpful in MLP layers
3. **Activation**: ReLU or SeLU
4. **Optimizer**: Adam or Adagrad
5. **Embedding Dim**: 32-64""",
        "dataset": "frappe",
        "init_params": """    embedding_dim=64,
    hidden_units=[128, 64],
    dropout=0.2,""",
    },
    "PNN": {
        "full_name": "Product-based Neural Network",
        "paper": "Qu et al. 2016 - Product-based Neural Networks for User Response Prediction",
        "intro": "uses a product layer to capture interactive patterns between inter-field categories before feeding into fully connected layers",
        "architecture": """**Product Layer:**

1. **Embedding Layer**: Feature embeddings
2. **Product Layer**:
   - Linear part (z): Concatenation
   - Quadratic part (p): Inner or Outer products of embeddings
3. **Hidden Layers**: MLP on top of product layer

**Key Innovation**: Explicitly modeling feature interactions via products (IPNN/OPNN)

**Architecture:**
```
Embeddings -> [Linear z, Product p] -> MLP -> Prediction
```""",
        "math": """**Inner Product (IPNN):**
```
p_ij = <v_i, v_j>
```

**Outer Product (OPNN):**
```
p_ij = v_i v_j^T
```

**Prediction:**
```
l_1 = ReLU(W_z z + W_p p + b_1)
ŷ = σ(W_out l_n + b_out)
```""",
        "use_cases": """✅ **Perfect For:**
- CTR prediction
- Multi-field categorical data
- When interactions are crucial
- Ad click prediction

❌ **Not For:**
- High-dimensional sparse data without fields
- Sequence modeling""",
        "best_practices": """1. **Inner vs Outer**: IPNN usually faster and sufficient
2. **Kernel Trick**: Use factorization to speed up product layer
3. **Hidden Units**: [256, 128, 64]
4. **Regularization**: L2 or Dropout needed""",
        "dataset": "criteo",
        "init_params": """    embedding_dim=64,
    use_inner_product=True,""",
    },
    # ========== BATCH 5: MULTI-TASK & ADVANCED ==========
    "MMoE": {
        "full_name": "Multi-gate Mixture-of-Experts",
        "paper": "Ma et al. 2018 - Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (Google)",
        "intro": "uses a Mixture-of-Experts (MoE) architecture with task-specific gating networks to learn task relationships automatically",
        "architecture": """**Multi-Task Learning with Gates:**

1. **Shared Experts**: Multiple expert networks share input
2. **Task-Specific Gates**: Learn how to weigh experts for each task
3. **Task Towers**: Specific networks for each objective (e.g., CTR, CVR)

**Key Innovation**: Solves "seesaw" problem where improving one task hurts another

**Architecture:**
```
Input -> [Expert 1] [Expert 2] ... [Expert N]
            |          |              |
         [Gate A]   [Gate B]       [Gate C]
            ↓          ↓              ↓
         [Tower A]  [Tower B]      [Tower C]
```""",
        "math": """**Mixture of Experts:**
```
f(x) = Σ_{i=1}^n g(x)_i · E_i(x)
```

**Gating Network (Task k):**
```
g^k(x) = softmax(W_g^k x)
```

**Final Output (Task k):**
```
y_k = h^k(f^k(x))
```
where h^k is the task tower.""",
        "use_cases": """✅ **Perfect For:**
- Multi-objective optimization (e.g., Clicks & Conversions)
- When tasks have complex relationships (correlated or conflicting)
- Large-scale production systems
- Reducing parameter count vs separate models

❌ **Not For:**
- Single task learning
- Small datasets
- When tasks are completely unrelated""",
        "best_practices": """1. **Number of Experts**: 4-8 usually sufficient
2. **Expert Size**: Smaller than single model
3. **Gate Bias**: Initialize to uniform
4. **Task Weights**: Tune loss weights (e.g., 1.0 for CTR, 2.0 for CVR)""",
        "dataset": "census-income",
        "init_params": """    num_experts=4,
    expert_dim=64,
    task_names=['income', 'marital'],""",
    },
    "PLE": {
        "full_name": "Progressive Layered Extraction",
        "paper": "Tang et al. 2020 - Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (Tencent)",
        "intro": "improves MMoE by explicitly separating shared and task-specific experts to avoid negative transfer",
        "architecture": """**CGC (Customized Gate Control):**

1. **Task-Specific Experts**: Only feed into specific task
2. **Shared Experts**: Feed into all tasks
3. **Multi-Level**: Stacked CGC modules for deep extraction

**Key Innovation**: Eliminates negative transfer by isolating task-specific knowledge

**Architecture:**
```
[Specific Exp A] [Shared Exp] [Specific Exp B]
       |              |              |
    [Gate A]          |           [Gate B]
       ↓              |              ↓
    [Tower A]         |           [Tower B]
```""",
        "math": """**CGC Aggregation:**
```
E^k = [SpecificExperts^k, SharedExperts]
y^k = Gate(x) · E^k(x)
```
Gate only selects from relevant experts + shared experts.

**Loss:**
```
L = Σ_k w_k L_k(y_k, label_k)
```""",
        "use_cases": """✅ **Perfect For:**
- Complex multi-task scenarios
- When MMoE suffers from negative transfer
- Highly conflicting tasks (e.g., Like vs Share)
- Deep multi-level representation learning

❌ **Not For:**
- Simple tasks
- Limited computational budget (more params than MMoE)""",
        "best_practices": """1. **Levels**: 1-2 CGC layers
2. **Expert Split**: 50% shared, 50% specific
3. **Gate Activation**: Softmax
4. **Gradient Analysis**: Monitor gradient conflicts""",
        "dataset": "census-income",
        "init_params": """    num_shared_experts=2,
    num_specific_experts=2,
    task_names=['income', 'marital'],""",
    },
    "Monolith": {
        "full_name": "Monolith: Real Time Recommendation System",
        "paper": "Liu et al. 2022 - Monolith: Real Time Recommendation System With Collisionless Embedding Table (ByteDance)",
        "intro": "designed for online training with collisionless embedding tables and dynamic feature eviction",
        "architecture": """**Online Learning Architecture:**

1. **Collisionless Embedding**: Hash-free, direct mapping
2. **Dynamic Eviction**: Remove stale features to save memory
3. **Online Training**: Updates in real-time streaming
4. **Parameter Server**: Distributed parameter synchronization

**Key Innovation**: Handling non-stationary distribution with real-time updates

**Architecture:**
```
Stream -> [Collisionless Embed] -> [Deep Network] -> Update
                  ↑
           [Eviction Policy]
```""",
        "math": """**Collisionless Hash:**
```
idx = Map(feature_value)  # No modulo collision
```

**Frequency Filter:**
Only create embedding if `count(feature) > threshold`.

**Adaptive Learning Rate:**
```
lr_t = lr_0 / sqrt(Σ g_t^2)
```""",
        "use_cases": """✅ **Perfect For:**
- Real-time recommendation (TikTok scale)
- Streaming data
- Non-stationary user interests
- Massive sparse features
- Low-latency updates

❌ **Not For:**
- Batch processing
- Static datasets
- Small scale systems""",
        "best_practices": """1. **Eviction**: Bloom filter for frequency
2. **Sync**: Async parameter updates
3. **Batch Size**: Small for online (or mini-batch)
4. **Fault Tolerance**: Snapshotting""",
        "dataset": "criteo-streaming",
        "init_params": """    embedding_dim=64,
    collisionless=True,
    eviction_threshold=10,""",
    },
    "TDM": {
        "full_name": "Tree-based Deep Model",
        "paper": "Zhu et al. 2018 - Learning Tree-based Deep Model for Recommender Systems (Alibaba)",
        "intro": "uses a hierarchical tree structure to index items, allowing logarithmic time retrieval with deep learning models",
        "architecture": """**Tree Indexing:**

1. **Item Tree**: Items are leaf nodes of a balanced tree
2. **Beam Search**: Traverse tree level-by-level
3. **Node Prediction**: Predict probability of user liking child nodes
4. **Deep Network**: Scores user-node pairs

**Key Innovation**: O(log N) retrieval with deep models (breaking vector search limit)

**Architecture:**
```
Level 1 -> Top-K Nodes -> Level 2 -> Top-K Nodes ... -> Items
```""",
        "math": """**Probability:**
```
P(u likes node n) = softmax(DNN(u, n))
```

**Max-Heap Property:**
```
P(parent) >= P(child)
```
Approximated by training.

**Retrieval:**
Beam search with width K at each level.""",
        "use_cases": """✅ **Perfect For:**
- Massive item corpus (billions)
- Replacing ANN (Approximate Nearest Neighbor)
- Full corpus retrieval
- E-commerce (Alibaba scale)

❌ **Not For:**
- Small item sets
- High-frequency updates to item set (tree rebuild)
- Simple ranking""",
        "best_practices": """1. **Tree Construction**: K-Means clustering on item embeddings
2. **Beam Size**: 20-50
3. **Tree Depth**: log_2(Items)
4. **Negative Sampling**: Sample from same level""",
        "dataset": "taobao-large",
        "init_params": """    tree_depth=10,
    beam_size=20,
    node_dim=64,""",
    },
}


# Helper function to generate model content
def get_model_content(model_name: str) -> dict:
    """Get detailed content for a model."""
    if model_name in MODEL_DATABASE:
        return MODEL_DATABASE[model_name]

    # Return generic content for models not yet detailed
    return {
        "full_name": model_name,
        "intro": f"is a recommendation model",
        "architecture": f"{model_name} architecture details.",
        "math": f"Mathematical formulation for {model_name}.",
        "use_cases": f"Use cases for {model_name}.",
        "best_practices": "1. Tune hyperparameters\n2. Use validation set",
        "dataset": "movielens-100k",
        "init_params": "    embedding_dim=64,",
    }
