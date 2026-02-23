# IMShow (Demo Frontends)

Instant demo frontend connector for showcasing recommendation models with themed UIs.

## Quick Start

```python
from corerec.imshow import connector

def my_recommender(user_id, num_items=10):
    return [{"id": str(i), "title": f"Item {i}"} for i in range(num_items)]

demo = connector(my_recommender, frontend="spotify", title="My Recommender")
demo.run()
```

## Available Frontends

- **spotify** — Dark green music streaming theme
- **youtube** — Red/white video platform theme
- **netflix** — Dark red movie/TV streaming theme

## API Reference

### Connector

```{eval-rst}
.. automodule:: corerec.imshow.connector
   :members:
   :show-inheritance:
```

### Frontends

```{eval-rst}
.. automodule:: corerec.imshow.frontends
   :members:
```

### Utilities

```{eval-rst}
.. automodule:: corerec.imshow.utils
   :members:
```
