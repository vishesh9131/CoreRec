# CoreRec Demo Frontends - Web Interface

🌐 **Modern Web Frontend for CoreRec Demo Platforms**

This is the new web-based frontend system for CoreRec demo platforms, providing authentic, professional interfaces that look and feel like real platforms (Spotify, YouTube, Netflix).

## ✨ Features

### 🎵 Platform-Authentic Interfaces
- **Spotify**: Dark theme, music player controls, playlists, recommendations
- **YouTube**: Video grid, channels, trending, search functionality  
- **Netflix**: Movie/TV show cards, genres, watchlist, ratings

### 🚀 Modern Technology Stack
- **Backend**: FastAPI with Python
- **Frontend**: Modern HTML5, CSS3, JavaScript (ES6+)
- **API**: RESTful endpoints for data and recommendations
- **Responsive**: Mobile-friendly design
- **Real-time**: Live updates and interactions

### 🎯 Key Improvements over Streamlit
- ✅ **Authentic Look**: Pixel-perfect platform replicas
- ✅ **Better Performance**: Faster loading and interactions
- ✅ **Mobile Support**: Responsive design for all devices
- ✅ **No Streamlit Errors**: Eliminates `set_page_config()` issues
- ✅ **Professional UX**: Smooth animations and transitions
- ✅ **Scalable**: Can handle many concurrent users

## 🚀 Quick Start

### Option 1: Run Both Frontend & Backend
```bash
cd corerec/demo_frontends
python launch.py
```

### Option 2: Using the Demo App
```bash
cd corerec/demo_frontends
python demo_app.py --mode web
```

### Option 3: Manual Setup
```bash
# Terminal 1: Start Backend
python backend/api.py

# Terminal 2: Start Frontend Server
cd web
python -m http.server 3000
```

## 📁 Project Structure

```
corerec/demo_frontends/
├── backend/
│   ├── api.py              # FastAPI backend server
│   └── models.py           # Data models and schemas
├── web/
│   ├── index.html          # Main platform selector
│   ├── css/
│   │   └── main.css        # Main styling
│   ├── js/
│   │   └── main.js         # Main JavaScript
│   └── platforms/
│       ├── spotify/
│       │   ├── index.html  # Spotify interface
│       │   ├── css/
│       │   │   └── spotify.css
│       │   └── js/
│       │       └── spotify.js
│       ├── youtube/
│       │   └── [YouTube files]
│       └── netflix/
│           └── [Netflix files]
├── launch.py               # Main launcher script
└── demo_app.py            # Enhanced demo app (Streamlit + Web)
```

## 🛠️ Installation

### Prerequisites
```bash
pip install fastapi uvicorn pandas numpy requests
```

### Optional (for development)
```bash
pip install pytest pytest-asyncio httpx  # For testing
```

## 📖 Usage

### 1. Basic Launch
```bash
python launch.py
```
- Starts backend on `http://localhost:8000`
- Starts frontend on `http://localhost:3000`
- Automatically opens browser

### 2. Advanced Options
```bash
# Backend only
python launch.py --backend-only

# Frontend only
python launch.py --frontend-only

# Custom port
python launch.py --port 8080

# No auto-browser
python launch.py --no-browser
```

### 3. Integration with Demo App
```bash
# Web mode
python demo_app.py --mode web

# Streamlit mode (default)
python demo_app.py --mode streamlit
```

## 🌐 API Endpoints

### Core Endpoints
- `GET /` - API status and info
- `GET /platforms` - List available platforms
- `POST /users/create` - Create new user session

### Platform Data
- `GET /platforms/{platform}/data` - Get platform content
- `GET /platforms/{platform}/recommendations` - Get recommendations
- `POST /platforms/{platform}/interactions` - Record user interactions

### User Management
- `GET /users/{user_id}/preferences` - Get user preferences
- `POST /users/{user_id}/preferences` - Update preferences
- `GET /users/{user_id}/history` - Get interaction history

## 🎨 Platform Details

### 🎵 Spotify Interface
- **Dark Theme**: Authentic Spotify green and dark styling
- **Music Player**: Play controls, progress bar, volume
- **Navigation**: Home, Search, Library sections
- **Content**: Track cards, playlists, recommendations
- **Interactions**: Like/unlike, add to playlist, play

### 📺 YouTube Interface
- **Red Theme**: YouTube's signature red and white design
- **Video Grid**: Thumbnail grid with view counts
- **Navigation**: Home, Trending, Subscriptions
- **Content**: Video cards, channels, categories
- **Interactions**: Like/dislike, subscribe, watch

### 🎬 Netflix Interface
- **Dark Theme**: Netflix's dark red styling
- **Movie/TV Grid**: Large content cards with ratings
- **Navigation**: Home, Movies, TV Shows, My List
- **Content**: Movies, TV shows, genres
- **Interactions**: Add to list, rate, watch

## 🔧 Configuration

### Environment Variables
```bash
export COREREC_API_HOST=localhost
export COREREC_API_PORT=8000
export COREREC_WEB_PORT=3000
export COREREC_DEBUG=true
```

### API Configuration
```python
# backend/api.py
app = FastAPI(
    title="CoreRec Demo Frontends API",
    version="1.0.0",
    debug=True  # Set to False in production
)
```

## 🧪 Testing

### Run API Tests
```bash
cd backend
python -m pytest tests/
```

### Manual Testing
1. Start the servers: `python launch.py`
2. Open http://localhost:3000
3. Click on a platform (Spotify, YouTube, Netflix)
4. Test interactions and recommendations

## 🚀 Deployment

### Local Development
```bash
python launch.py
```

### Production Deployment
```bash
# Using Gunicorn for production
pip install gunicorn
gunicorn backend.api:app --host 0.0.0.0 --port 8000

# Serve frontend with nginx or Apache
# Point document root to: corerec/demo_frontends/web/
```

### Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8000 3000

CMD ["python", "launch.py"]
```

## 🔍 Troubleshooting

### Common Issues

**❌ "FastAPI not found"**
```bash
pip install fastapi uvicorn
```

**❌ "CORS Error"**
- Ensure backend is running on port 8000
- Check browser console for specific errors

**❌ "No recommendations"**
- Verify sample data is generated
- Check API logs for errors

**❌ "Port already in use"**
```bash
# Use different ports
python launch.py --port 3001
```

### Debug Mode
```bash
# Enable debug logging
COREREC_DEBUG=true python launch.py
```

## 🆕 What's New vs Streamlit

| Feature | Streamlit | Web Frontend |
|---------|-----------|--------------|
| **Performance** | ⚠️ Slow reloads | ✅ Instant interactions |
| **Mobile** | ❌ Poor mobile UX | ✅ Responsive design |
| **Styling** | ⚠️ Limited customization | ✅ Full CSS control |
| **Errors** | ❌ `set_page_config()` errors | ✅ No Streamlit issues |
| **Scalability** | ⚠️ Single user focus | ✅ Multi-user ready |
| **Platform Look** | ⚠️ Generic Streamlit UI | ✅ Authentic platform UIs |

## 🤝 Contributing

1. **Add New Platform**: Create new directory in `web/platforms/`
2. **Enhance Features**: Modify existing platform interfaces
3. **API Extensions**: Add new endpoints in `backend/api.py`
4. **Testing**: Add tests in `backend/tests/`

## 📞 Support

- 📚 **Documentation**: See main CoreRec documentation
- 🐛 **Issues**: Create GitHub issues for bugs
- 💬 **Discussions**: Use GitHub discussions for questions
- 📧 **Contact**: Email support for urgent issues

---

**🎉 Enjoy your new professional demo frontends!**

The web interface provides a much more authentic and professional way to showcase your CoreRec recommendation models. No more Streamlit limitations - just beautiful, fast, responsive interfaces that look like the real platforms. 