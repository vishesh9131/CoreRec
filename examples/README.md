# CoreRec Examples Directory

Welcome to CoreRec examples! This directory contains comprehensive examples showcasing CoreRec's capabilities.

---

## 🌟 Featured Example: Instagram Reels Recommender

### **🎯 Production-Grade Meta-Scale System**

**File:** `instagram_reels_recommender.py`  
**Documentation:** `INSTAGRAM_REELS_README.md` | `INSTAGRAM_REELS_CASE_STUDY.md`

A complete Instagram Reels recommendation system demonstrating **ALL 10** CoreRec production features:

✅ Multi-modal deep learning (Video + Audio + Text + Metadata)  
✅ Sequential user behavior modeling (GRU + Attention)  
✅ Three-stage pipeline (Retrieval → Ranking → Re-ranking)  
✅ Fairness & diversity constraints  
✅ Production serving (REST API)  
✅ MLOps integration (MLflow tracking)  
✅ Complete serialization & versioning  

**Quick Start:**
```bash
python instagram_reels_recommender.py
```

**Key Stats:**
- 500K interactions processed
- 37K reels indexed
- 2.4M parameter model
- 100% creator diversity
- NDCG@10: 0.0804

---

## 📚 Example Categories

### 1. Quick Start Examples

**Get up and running in 5 minutes:**

| File | Description | Complexity |
|------|-------------|------------|
| `content_filter_quickstart.py` | Content-based filtering basics | ⭐ Beginner |
| `unionized_quickstart.py` | Unionized filter engine intro | ⭐⭐ Intermediate |
| `engines_quickstart.py` | Deep learning engines | ⭐⭐ Intermediate |

### 2. Content Filter Examples

**Content-based recommendation systems:**

| File | Algorithm | Use Case |
|------|-----------|----------|
| `content_filter_quickstart.py` | TF-IDF + Cosine Similarity | Document similarity |
| `content_filter_tfidf_example.py` | Advanced TF-IDF | News, articles |

**Directory:** `ContentFilterExamples/`
- Context-aware profiling
- Multi-modal examples
- CNN/DKN based systems
- Learning paradigms

### 3. Collaborative Filter Examples

**User-item interaction based systems:**

**Directory:** `CollaborativeFilterExamples/`

| Example | Model | Description |
|---------|-------|-------------|
| `ex_ncf.py` | Neural Collaborative Filtering | Deep CF with embeddings |
| `dlrm_eg.py` | DLRM | Facebook's production model |
| `spotify_recommender/` | Multi-model | Complete music system |

### 4. Neural Network Engines

**Deep learning based recommendations:**

| File | Model | Description |
|------|-------|-------------|
| `engines_deepfm_example.py` | DeepFM | Feature interactions |
| `engines_dcn_example.py` | Deep & Cross Network | Explicit + implicit features |
| `engines_sasrec_example.py` | SASRec | Self-attention for sequences |
| `engines_gnnrec_example.py` | GNN | Graph neural networks |
| `engines_nasrec_example.py` | NAS | Neural architecture search |
| `engines_mind_example.py` | MIND | Multi-interest network |

### 5. Unionized Filter Examples

**Hybrid and advanced algorithms:**

| File | Algorithm | Best For |
|------|-----------|----------|
| `unionized_fast_example.py` | FAST | Fast approximation |
| `unionized_sar_example.py` | SAR | Smart adaptive |
| `unionized_rbm_example.py` | RBM | Restricted Boltzmann |
| `unionized_rlrmc_example.py` | RLRMC | Robust low-rank |
| `unionized_geomlc_example.py` | GeoMLC | Geometric |

### 6. Advanced Examples

**Production-grade systems:**

| File | Focus | Level |
|------|-------|-------|
| `instagram_reels_recommender.py` | **Meta-scale system** | ⭐⭐⭐ Expert |
| `demo_frontends_example.py` | Web UI + Backends | ⭐⭐ Intermediate |
| `dien_example.py` | Deep Interest Evolution | ⭐⭐⭐ Expert |

### 7. YouTube MoE Example

**Directory:** `Youtube_MoE/`

Multi-task learning with Mixture of Experts:
- Data loader for MovieLens
- MoE architecture
- Multi-task optimization

### 8. Utility Examples

| File | Purpose |
|------|---------|
| `utils_example_data.py` | Synthetic data generation |
| `imshow_connector_example.py` | Visualization tools |
| `run_all_algo_tests_example.py` | Batch testing |

---

## 🚀 Running Examples

### Prerequisites

```bash
# Install CoreRec
pip install corerec

# Or from source
cd CoreRec/
pip install -e .
```

### Basic Usage

```bash
# Navigate to examples
cd examples/

# Run any example
python content_filter_quickstart.py
python engines_deepfm_example.py
python instagram_reels_recommender.py
```

### With Custom Data

```python
# Most examples accept custom data
from corerec.engines.unionizedFilterEngine.nn_base.ncf import NCF
import pandas as pd

# Your data
data = pd.DataFrame({
    'user_id': [...],
    'item_id': [...],
    'rating': [...]
})

# Train
model = NCF(name="my_model", embedding_dim=64)
model.fit(data)

# Recommend
recs = model.recommend(user_id=1, top_n=10)
```

---

## 📖 Learning Path

### For Beginners (New to RecSys)

1. **Start:** `content_filter_quickstart.py`
   - Learn: Basic similarity
   - Time: 10 minutes

2. **Next:** `unionized_quickstart.py`
   - Learn: Collaborative filtering
   - Time: 20 minutes

3. **Then:** `engines_quickstart.py`
   - Learn: Deep learning models
   - Time: 30 minutes

**Total:** ~1 hour to understand RecSys basics

### For Intermediate (Know RecSys Basics)

1. **Start:** `CollaborativeFilterExamples/ex_ncf.py`
   - Learn: Neural CF implementation
   - Time: 30 minutes

2. **Next:** `engines_deepfm_example.py`
   - Learn: Feature interactions
   - Time: 45 minutes

3. **Then:** `demo_frontends_example.py`
   - Learn: Full system with UI
   - Time: 1 hour

**Total:** ~2 hours to build production systems

### For Experts (Building at Scale)

1. **Study:** `instagram_reels_recommender.py`
   - Learn: Meta-scale architecture
   - Time: 2 hours

2. **Study:** `INSTAGRAM_REELS_CASE_STUDY.md`
   - Learn: Production deployment
   - Time: 1 hour

3. **Implement:** Your own system
   - Apply: All CoreRec features
   - Time: Varies

**Total:** Build world-class systems

---

## 🎯 Example by Use Case

### E-commerce Product Recommendations

**Best:** `engines_deepfm_example.py`
- Handles sparse features
- Feature interactions
- Cold start capable

```python
from corerec.engines.deepfm import DeepFM
model = DeepFM(field_dims=[1000, 500, 100])
model.fit(user_product_data)
```

### Video/Content Streaming

**Best:** `instagram_reels_recommender.py`
- Multi-modal features
- Sequential viewing patterns
- Diversity constraints

```python
model = InstagramReelsRecommender()
model.fit(viewing_data)
recs = model.recommend(user_id, top_k=10)
```

### Music/Playlist Recommendations

**Best:** `CollaborativeFilterExamples/spotify_recommender/`
- Audio features
- Session-based
- Sequence aware

### News/Article Recommendations

**Best:** `content_filter_tfidf_example.py`
- Text similarity
- Real-time updates
- Trending topics

### Social Network

**Best:** `engines_gnnrec_example.py`
- Graph structure
- Social signals
- Community detection

---

## 🔧 Modifying Examples

### Add Your Data

```python
# Replace synthetic data with yours
import pandas as pd

# Load your data
data = pd.read_csv('my_data.csv')

# Ensure columns match
assert 'user_id' in data.columns
assert 'item_id' in data.columns
assert 'rating' in data.columns  # or 'interaction'

# Run example as normal
model.fit(data)
```

### Adjust Parameters

```python
# Most models accept config dicts
model = NCF(
    name="my_ncf",
    embedding_dim=128,    # Increase for more capacity
    num_layers=4,         # Deeper network
    dropout=0.3,          # More regularization
    learning_rate=0.001,
    num_epochs=50
)
```

### Add Custom Features

```python
# Extend feature extractors
class MyFeatureExtractor(MultiModalFeatureExtractor):
    def extract_custom_features(self, item_id):
        # Your custom logic
        return custom_features

model = MyRecommender(feature_extractor=MyFeatureExtractor())
```

---

## 📊 Performance Tips

### Speed Up Training

```python
# Use GPU
model = model.cuda()

# Increase batch size
trainer = Trainer(model, batch_size=1024)

# Reduce epochs with early stopping
trainer.add_callback(EarlyStopping(patience=3))
```

### Improve Quality

```python
# Larger embeddings
model = NCF(embedding_dim=256)

# Deeper networks
model = DeepFM(mlp_dims=[512, 256, 128, 64])

# More training data
# Quality > quantity, but both matter

# Feature engineering
# Add domain-specific features
```

### Scale to Production

```python
# See instagram_reels_recommender.py for:
- Batch inference
- REST API serving  
- Model caching
- Distributed training
- Monitoring & logging
```

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Ensure CoreRec is installed
pip install corerec

# Or install from source
cd CoreRec/
pip install -e .
```

### Memory Errors

```python
# Reduce batch size
trainer = Trainer(model, batch_size=128)

# Reduce embedding dimension
model = NCF(embedding_dim=32)

# Use gradient checkpointing
model.enable_gradient_checkpointing()
```

### Slow Training

```python
# Use GPU
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Reduce model size
# Or use a simpler model for prototyping
```

---

## 📞 Support

**Documentation:** `/docs/`  
**Tests:** `/tests/`  
**Issues:** GitHub Issues  
**Email:** sciencely98@gmail.com

---

## 🎉 Featured Systems Built with CoreRec

### 1. Instagram Reels Recommender ⭐⭐⭐
- **Scale:** Millions of videos, billions of users
- **Latency:** <100ms target
- **Features:** Multi-modal, sequential, fair
- **File:** `instagram_reels_recommender.py`

### 2. Spotify Music Recommender ⭐⭐
- **Scale:** Audio library
- **Features:** Session-based, playlist generation
- **Directory:** `CollaborativeFilterExamples/spotify_recommender/`

### 3. Demo Frontends ⭐⭐
- **Platforms:** Netflix, Spotify, YouTube-style UIs
- **Features:** Full web apps with backends
- **File:** `demo_frontends_example.py`

---

## 🚀 Next Steps

1. **Try a quickstart:** Run `content_filter_quickstart.py`
2. **Explore your use case:** Find matching example above
3. **Study architecture:** Read `instagram_reels_recommender.py`
4. **Build your system:** Use CoreRec's full infrastructure
5. **Deploy to production:** Follow deployment guides

**Happy recommending!** 🎯

---

**CoreRec Version:** v0.5.2.0+  
**Last Updated:** October 12, 2025  
**Maintainer:** Vishesh Yadav (sciencely98@gmail.com)

