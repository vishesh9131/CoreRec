# Demo Frontends Tutorial

CoreRec's IMShow module lets you plug any recommendation function into themed frontend UIs for instant demos.

## Quick Start

```python
from corerec.imshow import connector

def my_recommender(user_id, num_items=10):
    """Your recommendation function."""
    return [
        {"id": str(i), "title": f"Song {i}", "artist": f"Artist {i}"}
        for i in range(num_items)
    ]

demo = connector(my_recommender, frontend="spotify", title="My Music App")
demo.run()  # Opens browser at http://localhost:8080
```

## Available Frontends

### Spotify Theme (Music)

```python
from corerec.imshow import spotify_demo

demo = spotify_demo(my_recommender, title="Music Discovery")
demo.run()
```

Required fields: `id`, `title`, `artist`
Optional: `album`, `genre`, `duration`, `popularity`

### YouTube Theme (Video)

```python
from corerec.imshow import youtube_demo

demo = youtube_demo(my_video_recommender, title="Video Recommendations")
demo.run()
```

Required fields: `id`, `title`, `channel`
Optional: `category`, `duration`, `views`, `likes`

### Netflix Theme (Movies/TV)

```python
from corerec.imshow import netflix_demo

demo = netflix_demo(my_movie_recommender, title="Movie Night")
demo.run()
```

Required fields: `id`, `title`, `genre`
Optional: `year`, `rating`, `duration`, `description`

## Function Signature

Your recommendation function should accept:
- `user_id` (str): User identifier
- `num_items` (int): Number of items to return (default: 10-12)

And return a list of dictionaries, each with at least `id` and `title`.

## Advanced Usage

### Background Mode

```python
url = demo.run(background=True)
print(f"Demo running at {url}")

# Do other work...

demo.stop()
```

### Recording Interactions

```python
interactions = demo.get_interactions()
demo.export_interactions("my_interactions.json")
```

### Custom Ports

```python
demo = connector(
    my_recommender,
    frontend="spotify",
    port=9000,
    api_port=9001,
)
```
