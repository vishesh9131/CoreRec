# CoreRec IMShow - Instant Demo Frontend Connector

> **Plug and play your recommendation system into beautiful frontends in 3 lines of code!**

CoreRec IMShow is a mini-framework that allows researchers, students, and developers to instantly connect their recommendation models to professional-looking frontend interfaces. Perfect for demos, presentations, and prototyping.

## 🚀 Quick Start

```python
import corerec.imshow as ii

# Your recommendation function
def my_recommender(user_id, num_items=10):
    # Your recommendation logic here
    return [
        {"id": "1", "title": "Song 1", "artist": "Artist 1"},
        {"id": "2", "title": "Song 2", "artist": "Artist 2"},
    ]

# Plug into a beautiful frontend
demo = ii.connector(my_recommender, frontend="spotify", title="My Music Recommender")
demo.run()  # 🎉 Opens browser with Spotify-style interface!
```

## 🎨 Available Frontends

### 🎵 Spotify (Music)
- **Theme**: Dark green, modern music interface
- **Use case**: Music recommendations, playlists, audio features
- **Required fields**: `id`, `title`, `artist`
- **Optional fields**: `album`, `genre`, `duration`, `popularity`

### 📺 YouTube (Videos)  
- **Theme**: Red and white, video platform interface
- **Use case**: Video recommendations, content creators, trending
- **Required fields**: `id`, `title`, `channel`
- **Optional fields**: `category`, `duration`, `views`, `likes`

### 🎬 Netflix (Movies/TV)
- **Theme**: Dark red, streaming platform interface  
- **Use case**: Movie/TV recommendations, content rating, genres
- **Required fields**: `id`, `title`, `genre`
- **Optional fields**: `year`, `rating`, `duration`, `description`

## 📦 Installation

```bash
# Install CoreRec (if not already installed)
pip install corerec

# IMShow is included - no additional installation needed!
```

## 🔧 Basic Usage

### Simple Example

```python
import corerec.imshow as ii

def simple_recommender(user_id, num_items=10):
    """Simple recommendation function."""
    recommendations = []
    for i in range(num_items):
        recommendations.append({
            "id": f"song_{i}",
            "title": f"Great Song {i+1}",
            "artist": f"Artist {i+1}"
        })
    return recommendations

# Create and run demo
demo = ii.connector(
    predict_function=simple_recommender,
    frontend="spotify",
    title="My Music Recommender",
    description="Personalized music recommendations"
)

demo.run()
```

### Different Parameter Names

The system automatically maps common parameter names:

```python
def my_video_recommender(user_id, k=12):  # Using 'k' instead of 'num_items'
    """Your function can use any parameter names."""
    return [{"id": f"video_{i}", "title": f"Video {i+1}", "channel": "MyChannel"} for i in range(k)]

demo = ii.connector(my_video_recommender, frontend="youtube")
demo.run()
```

### Background Mode

```python
# Run in background (non-blocking)
demo = ii.connector(my_recommender, frontend="netflix")
url = demo.run(background=True)

print(f"Demo running at {url}")
# Your script continues...

# Stop later
demo.stop()
```

## 🎯 Real-World Integration

### With CoreRec Models

```python
import corerec.imshow as ii
from your_corerec_model import MyModel  # Your trained model

# Load your trained CoreRec model
model = MyModel.load("path/to/model.pkl")

def corerec_recommender(user_id, num_items=10):
    """Wrapper for your CoreRec model."""
    # Call your actual model
    recommendations = model.predict(user_id, k=num_items)
    
    # Return in the expected format
    return recommendations

# Plug into frontend
demo = ii.connector(
    predict_function=corerec_recommender,
    frontend="spotify",
    title="My CoreRec Music Model",
    description="Advanced neural collaborative filtering"
)

demo.run()
```

### Custom Configuration

```python
demo = ii.connector(
    predict_function=my_recommender,
    frontend="spotify",
    title="Advanced Music AI",
    description="Powered by transformer-based embeddings",
    port=8080,           # Custom port
    debug=True,          # Enable debug mode
    auto_open=False      # Don't auto-open browser
)

demo.run()
```

## 📊 Monitoring & Analytics

### Recording Interactions

```python
demo = ii.connector(my_recommender, frontend="spotify")
url = demo.run(background=True)

# Users interact with the interface...
# Interactions are automatically recorded

# Get interaction data
interactions = demo.get_interactions()
print(f"Recorded {len(interactions)} interactions")

# Export to JSON
demo.export_interactions("my_demo_data.json")
```

### Real-time Stats

```python
# Get demo info and stats
info = demo.get_info()
print(f"Total interactions: {info['stats']['total_interactions']}")
print(f"Unique users: {info['stats']['unique_users']}")
```

## 🛠 Advanced Features

### Multiple Frontends

```python
# Run multiple demos simultaneously
music_demo = ii.connector(music_recommender, "spotify", port=8080)
video_demo = ii.connector(video_recommender, "youtube", port=8081) 
movie_demo = ii.connector(movie_recommender, "netflix", port=8082)

# Start all in background
music_demo.run(background=True)
video_demo.run(background=True)
movie_demo.run(background=True)
```

### Custom Data Formatting

```python
def flexible_recommender(user_id, num_items=10):
    """Your function can return any format."""
    # Can return list of dicts, list of strings, or any format
    return [
        {"custom_id": f"item_{i}", "name": f"Item {i}", "score": 0.9},
        # IMShow automatically formats for the frontend
    ]

demo = ii.connector(flexible_recommender, frontend="spotify")
demo.run()
```

## 🎪 Convenience Functions

### Quick Demos

```python
import corerec.imshow as ii

# List available frontends
ii.list_frontends()

# Launch quick demo with sample data
ii.quick_demo("spotify")

# Validate your function
ii.validate_function(my_recommender)

# Frontend-specific shortcuts
demo = ii.spotify_demo(my_music_function, title="My Music AI")
demo = ii.youtube_demo(my_video_function, title="My Video AI")
demo = ii.netflix_demo(my_movie_function, title="My Movie AI")
```

## 📋 Function Requirements

Your recommendation function should:

1. **Accept parameters** like `user_id`, `num_items`, `k`, etc. (flexible parameter names)
2. **Return a list** of recommendations
3. **Each recommendation** should be a dictionary with at least:
   - `id`: Unique identifier
   - `title` or `name`: Display name
   - Frontend-specific fields (see frontend docs above)

### Example Function Signatures

```python
# Any of these work:
def recommender(user_id, num_items=10): ...
def recommender(user_id, k=10): ...
def recommender(user_id, n_items=10): ...
def recommender(user_id, top_k=10): ...
def recommender(user_id): ...  # Will use defaults
def recommender(): ...  # For simple cases
```

## 🚨 Error Handling

IMShow gracefully handles errors:

```python
def sometimes_fails(user_id, num_items=10):
    if user_id == "bad_user":
        raise Exception("User not found!")
    return [{"id": "1", "title": "Song 1", "artist": "Artist 1"}]

# If your function fails, IMShow shows sample data instead
demo = ii.connector(sometimes_fails, frontend="spotify", debug=True)
demo.run()  # Will show error in debug mode, but continue working
```

## 📚 Examples

See `examples.py` for comprehensive examples:

```bash
python -m corerec.imshow.examples
```

### Example Categories:

1. **Simple Music Recommender** - Basic Spotify interface
2. **Video Recommender** - YouTube interface with different parameters  
3. **Netflix Movies** - Movie/TV recommendations
4. **CoreRec Integration** - Real model integration
5. **Background Mode** - Non-blocking demos with interaction monitoring
6. **Multiple Frontends** - Running multiple demos simultaneously

## 🔧 Configuration Options

### Connector Parameters

```python
ii.connector(
    predict_function=my_func,     # Required: Your recommendation function
    frontend="spotify",           # Required: Frontend type
    title="My Demo",             # Optional: Demo title
    description="Description",    # Optional: Demo description  
    port=8080,                   # Optional: Frontend port (auto-assigned)
    api_port=9080,               # Optional: API port (auto-assigned)
    debug=False,                 # Optional: Enable debug logging
    auto_open=True,              # Optional: Auto-open browser
    **kwargs                     # Optional: Additional config
)
```

### Run Parameters

```python
demo.run(
    background=False,            # Run in background (non-blocking)
    open_browser=None           # Override auto_open setting
)
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
```python
# Use auto port assignment
demo = ii.connector(my_func, frontend="spotify")  # Auto-assigns ports

# Or specify custom port
demo = ii.connector(my_func, frontend="spotify", port=8081)
```

**Function not compatible:**
```python
# Validate your function first
ii.validate_function(my_func)

# Enable debug mode to see what's happening
demo = ii.connector(my_func, frontend="spotify", debug=True)
```

**Missing dependencies:**
```python
# IMShow works with basic Python
# For advanced features, install:
pip install fastapi uvicorn  # Better API server
```

## 🎨 Customization

### Frontend Themes

Each frontend comes with authentic styling:
- **Spotify**: Dark theme with green accents
- **YouTube**: Light theme with red accents  
- **Netflix**: Dark theme with red accents

### Data Fields

IMShow automatically:
- Maps your data to frontend requirements
- Adds missing fields with sensible defaults
- Formats data for optimal display

## 🚀 Performance

- **Lightweight**: No heavy dependencies
- **Fast**: Minimal overhead on your functions
- **Scalable**: Handles multiple concurrent users
- **Fallbacks**: Graceful error handling with sample data

## 🤝 Contributing

Want to add more frontends? The system is extensible:

1. Add frontend config to `frontends.py`
2. Create HTML template with your styling
3. Add to available frontends list

## 📜 License

Same as CoreRec main project.

## 🙋 Support

- **Issues**: Report bugs in the main CoreRec repository
- **Questions**: Use discussions for help
- **Examples**: Check `examples.py` for comprehensive usage

---

**Happy recommending! 🎯✨**

*Make your recommendation systems shine with beautiful, professional frontends in minutes, not hours.* 