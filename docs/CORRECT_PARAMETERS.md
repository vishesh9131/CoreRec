# CoreRec Model Parameter Quick Reference

**CRITICAL**: Always check the actual `__init__` signature! Documentation may be outdated.

## SASRec (Self-Attentive Sequential Recommendation)

### ✅ CORRECT Parameters:
```python
from corerec.engines.sasrec import SASRec

model = SASRec(
    name="SASRec",
    hidden_units=64,          # Embedding/hidden dimension
    num_blocks=2,             # Number of transformer blocks  
    num_heads=1,              # Number of attention heads
    dropout_rate=0.1,         # Dropout rate
    max_seq_length=50,        # Maximum sequence length
    position_encoding="learned",  # "learned" or "sinusoidal"
    attention_type="causal",  # "causal" or "full"
    activation="gelu",        # Activation function
    learning_rate=1e-3,
    batch_size=128,
    num_epochs=10,
    verbose=True
)
```

### ❌ WRONG (from old docs):
```python
# DON'T USE THESE - THEY'RE WRONG!
embedding_dim=64,  # ❌ Use hidden_units
n_layers=2,        # ❌ Use num_blocks  
n_heads=2,         # ❌ Use num_heads
max_len=50,        # ❌ Use max_seq_length
dropout=0.1,       # ❌ Use dropout_rate
epochs=10,         # ❌ Use num_epochs
```

## DCN (Deep & Cross Network)

### ✅ CORRECT Parameters:
```python
from corerec.engines.dcn import DCN

model = DCN(
    name="DCN",
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64],
    dropout=0.2,
    learning_rate=0.001,
    batch_size=256,
    epochs=20,
    verbose=False
)
```

## DeepFM

### ✅ CORRECT Parameters:
```python
from corerec.engines.deepfm import DeepFM

model = DeepFM(
    name="DeepFM",
    embedding_dim=32,
    deep_layers=[256, 128, 64],
    dropout=0.3,
    use_bn=True,
    learning_rate=0.001,
    batch_size=512,
    epochs=20,
    verbose=False
)
```

## GNNRec (Graph Neural Network)

### ✅ CORRECT Parameters:
```python
from corerec.engines.gnnrec import GNNRec

model = GNNRec(
    name="GNNRec",
    embedding_dim=128,
    num_layers=3,
    aggregator='mean',  # 'mean', 'sum', 'max', 'attention'
    dropout=0.1,
    learning_rate=0.001,
    batch_size=256,
    epochs=20,
    verbose=False
)
```

## How to Find Correct Parameters

### Method 1: Check Source Code
```python
import inspect
from corerec.engines.sasrec import SASRec

# View all parameters
sig = inspect.signature(SASRec.__init__)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else 'REQUIRED'
        print(f"{param_name}: {default}")
```

### Method 2: Use help()
```python
from corerec.engines.sasrec import SASRec
help(SASRec.__init__)
```

### Method 3: Check the __init__ docstring
```python
from corerec.engines.sasrec import SASRec
print(SASRec.__init__.__doc__)
```

## Common Traps

1. **SASRec**: Uses `hidden_units` NOT `embedding_dim`
2. **Parameter names vary**: Some use `epochs`, others use `num_epochs`
3. **Dropout naming**: Some use `dropout`, others use `dropout_rate`
4. **Always verify**: Check the source before trusting docs!

## Report Documentation Bugs

If you find incorrect documentation, please:
1. Open an issue on GitHub: https://github.com/vishesh9131/CoreRec/issues
2. Include the model name and incorrect parameter names
3. We'll fix it ASAP!
