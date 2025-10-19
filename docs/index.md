# CoreRec Documentation

<div align="center">
    <img src="https://raw.githubusercontent.com/vishesh9131/CoreRec/main/docs/images/coreRec.svg" width="120" height="120">
    <h1>CoreRec & VishGraphs</h1>
    <p><strong>Advanced Recommendation Systems Library</strong></p>
</div>

[![Downloads](https://static.pepy.tech/badge/corerec)](https://pepy.tech/project/corerec)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/vishesh9131/corerec)
![PyPI - Downloads](https://img.shields.io/pypi/dm/corerec?label=PyPi%20Downloads)

---

## Welcome to CoreRec

CoreRec is a comprehensive, production-ready recommendation systems library that provides state-of-the-art algorithms, neural network architectures, and utilities for building powerful recommendation engines. Whether you're working with collaborative filtering, content-based filtering, or deep learning models, CoreRec has you covered.

## 🌟 Key Features

### 🎯 **Three Powerful Engines**

=== "Unionized Filter Engine"
    
    Collaborative filtering and hybrid methods
    
    - **Matrix Factorization**: SVD, ALS, NMF, PMF
    - **Neural Networks**: NCF, DeepFM, AutoInt, DCN
    - **Graph-Based**: LightGCN, DeepWalk, GNN
    - **Attention Mechanisms**: SASRec, Transformers
    - **Bayesian Methods**: BPR, Bayesian MF
    - **Sequential Models**: LSTM, GRU, Caser
    - **Variational Encoders**: VAE, CVAE
    
=== "Content Filter Engine"
    
    Content-based and feature-rich recommendations
    
    - **Traditional ML**: TF-IDF, SVM, Decision Trees, LightGBM
    - **Neural Networks**: DSSM, MIND, TDM, YouTube DNN, Transformers
    - **Graph-Based**: GNN, Semantic Models
    - **Embedding Learning**: Word2Vec, Doc2Vec
    - **Hybrid & Ensemble**: Attention, Stacking
    - **Fairness & Explainability**: Fair ranking, LIME
    - **Learning Paradigms**: Transfer, Meta, Few-shot, Zero-shot
    
=== "Deep Learning Models"
    
    State-of-the-art deep learning architectures
    
    - **DCN** (Deep & Cross Network)
    - **DeepFM** (Factorization Machines)
    - **GNNRec** (Graph Neural Networks)
    - **MIND** (Multi-Interest Network)
    - **NASRec** (Neural Architecture Search)
    - **SASRec** (Self-Attentive Sequential)

### 🏗️ **Modular Architecture**

- **Towers**: User & Item encoding towers (MLP, CNN, Transformer, Fusion)
- **Encoders**: Feature encoding and embedding layers
- **Losses**: Multiple loss functions (BCE, MSE, Triplet, BPR)
- **Optimizers**: Built-in optimizers (Adam, Nadam, RMSprop, SGD, etc.)

### 📊 **Complete ML Pipeline**

- **Data Loading & Preprocessing**: Format conversion, feature engineering
- **Training Pipeline**: Distributed training, hyperparameter tuning
- **Evaluation Metrics**: Precision, Recall, NDCG, MRR, Hit Rate
- **Model Serving**: Production-ready deployment utilities
- **Visualization**: Graph visualization with VishGraphs

### 🎨 **Visualization Suite (VishGraphs)**

- 2D and 3D graph visualization
- Interactive graph exploration
- Bipartite relationship visualization
- Network analysis tools

## 🚀 Quick Start

Get started with CoreRec in just a few lines of code:

```python
# Install CoreRec
pip install --upgrade corerec

# Example: Using Deep & Cross Network (DCN)
from corerec.engines.dcn import DCN

# Initialize model
model = DCN(
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64, 32],
    epochs=10
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=123, top_n=10)
print(recommendations)
```

## 📚 What's in the Documentation?

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Installation, quick start guides, and architecture overview
    
    [:octicons-arrow-right-24: Get Started](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Comprehensive guides on data preparation, training, and predictions
    
    [:octicons-arrow-right-24: Learn More](user-guide/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Detailed API documentation for all classes and methods
    
    [:octicons-arrow-right-24: Browse API](api/index.md)

-   :material-engine:{ .lg .middle } **Engines**

    ---

    In-depth documentation of all recommendation engines
    
    [:octicons-arrow-right-24: Explore Engines](engines/index.md)

-   :material-brain:{ .lg .middle } **Core Components**

    ---

    Towers, encoders, embeddings, and loss functions
    
    [:octicons-arrow-right-24: Core Docs](core/index.md)

-   :material-school:{ .lg .middle } **Examples**

    ---

    Real-world examples and use cases
    
    [:octicons-arrow-right-24: View Examples](examples/index.md)

-   :material-test-tube:{ .lg .middle } **Testing**

    ---

    Testing guidelines and test suite documentation
    
    [:octicons-arrow-right-24: Testing Guide](testing/index.md)

-   :material-account-group:{ .lg .middle } **Contributing**

    ---

    Contributing guidelines and development setup
    
    [:octicons-arrow-right-24: Contribute](contributing/index.md)

</div>

## 🎯 Use Cases

CoreRec is perfect for:

- **E-commerce**: Product recommendations, personalized shopping
- **Media & Entertainment**: Movie, music, video recommendations
- **Social Networks**: Friend suggestions, content curation
- **News & Articles**: Content discovery, personalized feeds
- **Education**: Course recommendations, learning paths
- **Research**: Academic research on recommendation systems

## 🏆 Why Choose CoreRec?

| Feature | CoreRec |
|---------|---------|
| **Algorithms** | 100+ state-of-the-art algorithms |
| **Flexibility** | Modular architecture, easy to extend |
| **Performance** | Optimized for speed and scalability |
| **Production-Ready** | Battle-tested, production deployment support |
| **Documentation** | Comprehensive docs with examples |
| **Active Development** | Regular updates and improvements |
| **Community** | Growing community of contributors |
| **Research-Friendly** | Implements latest research papers |

## 📈 Statistics

- **100+** Recommendation algorithms
- **1000+** Downloads per month
- **50+** Example scripts
- **100%** Test coverage for core components
- **MIT-style** Open source license

## 🤝 Community & Support

- **GitHub**: [vishesh9131/CoreRec](https://github.com/vishesh9131/CoreRec)
- **PyPI**: [corerec](https://pypi.org/project/corerec/)
- **Issues**: [Report bugs or request features](https://github.com/vishesh9131/CoreRec/issues)
- **Email**: sciencely98@gmail.com

## 📝 License

CoreRec is distributed for **research purposes only**. Please do not use it commercially without the author's consent.

---

<div align="center">
    <p><strong>Ready to build amazing recommendation systems?</strong></p>
    <p><a href="getting-started/installation/">Get Started Now →</a></p>
</div>


