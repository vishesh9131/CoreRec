# Spotify Music Recommender

A text-based music recommendation system built using the CoreRec framework, featuring:

- Content-based recommendations using TF-IDF on lyrics
- Collaborative filtering using DLRM (Deep Learning Recommendation Model)
- Flask web interface with real-time recommendations

## Running the Application

### From the Project Root Directory

For macOS/Linux users:
```bash
./run_spotify_recommender.sh
```

For Windows users:
```
run_spotify_recommender.bat
```

### From the Spotify Recommender Directory

If you're already in the spotify_recommender directory:
```bash
python app.py
```

The application will start on port 5003. Access it in your browser at:
```
http://localhost:5003
```

## Features

- Personalized music recommendations based on liked songs
- Text-based interface (no images) for lightweight performance
- Similar songs recommendations based on lyrics content
- Artist pages showing all songs by an artist
- Search functionality for finding specific songs or artists
- Real-time recommendations when liking/unliking songs

## How to Use

1. **Home Page**: Displays recommended songs, trending songs, and popular artists
2. **Song Page**: Shows lyrics, similar songs, and more songs by the same artist
3. **Artist Page**: Displays all songs from the selected artist
4. **Like Songs**: Click the heart icon to like a song and get instant recommendations
5. **Switch Users**: The app supports multiple test users; switch between them using the sidebar

## Technical Details

The recommender system uses a hybrid approach:
- Content-based filtering analyzes song lyrics to find similar songs
- DLRM collaborative filtering learns patterns from user-song interactions
- Weights are applied to combine both methods for better recommendations 