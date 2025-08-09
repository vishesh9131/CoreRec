import os
import json
import random
from pathlib import Path
import sys
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# Import the recommender models
from music_recommender import HybridMusicRecommender, DLRMRecommender

app = Flask(__name__)
app.secret_key = 'spotify_recommender_secret_key'

# Global variables
DATA_DIR = str(Path(__file__).parent.parent.parent.parent / "src/SANDBOX/dataset/spotify")
MODEL_PATH = str(Path(__file__).parent / "spotify_recommender_model.pkl")
DEFAULT_USER_ID = 1  # Default user ID

# Initialize the music recommender model
def get_recommender():
    try:
        # Try to load pre-trained model
        print(f"Attempting to load model from {MODEL_PATH}")
        if Path(MODEL_PATH).exists():
            recommender = HybridMusicRecommender.load(MODEL_PATH)
            print("Model loaded successfully")
            return recommender
        else:
            print("Model file not found, training a new model...")
            data_path = f"{DATA_DIR}/spotify_millsongdata.csv"
            
            # Create and train the model
            recommender = HybridMusicRecommender(name="SpotifyRecommender", dlrm_weight=0.4)
            recommender.fit(data_path)
            
            # Save the model
            recommender.save(MODEL_PATH)
            print("New model trained and saved")
            
            return recommender
    except Exception as e:
        print(f"Error loading or training recommender model: {e}")
        print("Using random recommendations as fallback")
        return None

# Get sample artists for display
def get_sample_artists(recommender, n_artists=8):
    if recommender and recommender.artist_data is not None:
        # Get most popular artists by song count
        popular_artists = recommender.artist_data.nlargest(n_artists, 'song_count')
        return [{'name': artist, 'song_count': count} 
                for artist, count in zip(popular_artists['artist'], popular_artists['song_count'])]
    return []

# Initialize global data
recommender = get_recommender()
sample_artists = get_sample_artists(recommender)

# Routes
@app.route('/')
def index():
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    
    # Get user's liked songs
    liked_songs = get_user_liked_songs(user_id)
    
    # Handle the case when recommender is None
    if recommender is None:
        # Provide some dummy recommendations when the model failed to load
        dummy_recommendations = get_dummy_recommendations(8)
        return render_template('index.html', 
                          user_id=user_id,
                          liked_songs=liked_songs,
                          recommendations=dummy_recommendations,
                          trending=get_trending_songs(5),
                          new_releases=get_new_releases(5),
                          sample_artists=sample_artists)
    
    # Get personalized recommendations if user has liked songs
    if liked_songs:
        recommendations = recommender.recommend(user_id, n_recommendations=8)
    else:
        # Get popular songs if no preferences
        recommendations = recommender.get_popular_songs(n_songs=8)
    
    # Get trending songs (random sample for demo)
    trending = get_trending_songs(5)
    
    # Get new releases (random sample for demo)
    new_releases = get_new_releases(5)
    
    return render_template('index.html', 
                          user_id=user_id,
                          liked_songs=liked_songs,
                          recommendations=recommendations,
                          trending=trending,
                          new_releases=new_releases,
                          sample_artists=sample_artists)

@app.route('/song/<song_key>')
def song_detail(song_key):
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    
    # URL decode the song key
    song_key = song_key.replace('_and_', '&')
    
    # Check if recommender is available
    if recommender is None:
        return redirect(url_for('index'))
    
    # Find the song in the recommender's data
    song = None
    if song_key in recommender.song_index_mapping:
        song_id = recommender.song_index_mapping[song_key]
        song_df = recommender.song_data[recommender.song_data['song_id'] == song_id]
        if not song_df.empty:
            song = song_df.iloc[0]
            song = {
                'artist': song['artist'],
                'song': song['song'],
                'song_key': song_key,
                'link': song['link'],
                'lyrics': song['text'],
                'is_liked': is_song_liked(user_id, song_key)
            }
    
    if not song:
        return redirect(url_for('index'))
    
    # Get similar songs
    similar_songs = recommender.get_similar_songs(song_key, n_similar=5)
    
    # Get more songs by the same artist
    more_from_artist = get_more_from_artist(song['artist'], exclude_song=song_key, n_songs=4)
    
    return render_template('song.html', 
                          user_id=user_id,
                          song=song,
                          similar_songs=similar_songs,
                          more_from_artist=more_from_artist,
                          sample_artists=sample_artists)

@app.route('/artist/<artist_name>')
def artist_page(artist_name):
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    
    # URL decode the artist name
    artist_name = artist_name.replace('_and_', '&')
    
    # Check if recommender is available
    if recommender is None:
        return redirect(url_for('index'))
    
    # Get artist songs
    artist_songs = []
    artist_df = recommender.song_data[recommender.song_data['artist'] == artist_name]
    for _, song in artist_df.iterrows():
        song_key = f"{song['artist']} - {song['song']}"
        artist_songs.append({
            'artist': song['artist'],
            'song': song['song'],
            'song_key': song_key,
            'link': song['link'],
            'is_liked': is_song_liked(user_id, song_key)
        })
    
    if not artist_songs:
        return redirect(url_for('index'))
    
    # Get similar artists (random for demo)
    similar_artists = sample_artists[:4]
    
    return render_template('artist.html', 
                          user_id=user_id,
                          artist_name=artist_name,
                          artist_songs=artist_songs,
                          similar_artists=similar_artists,
                          sample_artists=sample_artists)

@app.route('/search')
def search():
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    
    # Get search query
    query = request.args.get('q', '')
    if not query:
        return redirect(url_for('index'))
    
    # Check if recommender is available
    if recommender is None:
        return render_template('search.html', 
                             user_id=user_id,
                             query=query,
                             results=[],
                             sample_artists=sample_artists)
    
    # Search songs
    results = recommender.search_songs(query, n_results=20)
    # Add liked status to results
    for song in results:
        song['is_liked'] = is_song_liked(user_id, song['song_key'])
    
    return render_template('search.html', 
                          user_id=user_id,
                          query=query,
                          results=results,
                          sample_artists=sample_artists)

@app.route('/liked')
def liked_songs():
    # Get the current user from session or use default
    user_id = session.get('user_id', DEFAULT_USER_ID)
    
    # Get user's liked songs
    liked_songs = get_user_liked_songs(user_id)
    
    return render_template('liked.html', 
                          user_id=user_id,
                          liked_songs=liked_songs,
                          sample_artists=sample_artists)

@app.route('/switch_user/<int:user_id>')
def switch_user(user_id):
    session['user_id'] = user_id
    return redirect(request.referrer or url_for('index'))

@app.route('/api/like_song', methods=['POST'])
def like_song():
    user_id = session.get('user_id', DEFAULT_USER_ID)
    song_key = request.json.get('song_key', '')
    
    if not song_key:
        return jsonify({'success': False, 'error': 'No song key provided'})
    
    # Check if recommender is available
    if recommender is None:
        return jsonify({'success': False, 'error': 'Recommender system is not available'})
    
    # Like the song
    if song_key in recommender.song_index_mapping:
        recommender.add_user_preference(user_id, song_key, rating=1.0)
        session[f'user_{user_id}_likes'] = list(get_user_liked_songs(user_id))
        
        # Generate real-time recommendations based on the newly liked song
        real_time_recommendations = recommender.get_real_time_recommendations(user_id, n_recommendations=5)
        
        return jsonify({
            'success': True, 
            'recommendations': real_time_recommendations
        })
    
    return jsonify({'success': False, 'error': 'Song not found'})

@app.route('/api/unlike_song', methods=['POST'])
def unlike_song():
    user_id = session.get('user_id', DEFAULT_USER_ID)
    song_key = request.json.get('song_key', '')
    
    # Check if recommender is available
    if not song_key or recommender is None:
        return jsonify({'success': False, 'error': 'Invalid request or recommender not available'})
    
    # Remove the song from user preferences
    if song_key in recommender.song_index_mapping:
        song_id = recommender.song_index_mapping[song_key]
        if user_id in recommender.user_preferences and song_id in recommender.user_preferences[user_id]:
            del recommender.user_preferences[user_id][song_id]
            session[f'user_{user_id}_likes'] = list(get_user_liked_songs(user_id))
            
            # Get updated recommendations after unliking
            updated_recommendations = recommender.recommend(user_id, n_recommendations=5)
            
            return jsonify({
                'success': True,
                'recommendations': updated_recommendations
            })
    
    return jsonify({'success': False, 'error': 'Failed to unlike song'})

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """API endpoint to get recommendations for a user"""
    # Check if recommender is available
    if recommender is None:
        return jsonify({
            'success': False, 
            'error': 'Recommender system is not available',
            'recommendations': get_dummy_recommendations(5)
        })
    
    recommendations = recommender.recommend(user_id, n_recommendations=5)
    return jsonify({'success': True, 'recommendations': recommendations})

@app.route('/api/real_time_recommendations', methods=['POST'])
def real_time_recommendations():
    """API endpoint to get real-time recommendations based on current state"""
    user_id = request.json.get('user_id', DEFAULT_USER_ID)
    
    # Check if recommender is available
    if recommender is None:
        return jsonify({
            'success': False, 
            'error': 'Recommender system is not available',
            'recommendations': get_dummy_recommendations(5)
        })
    
    # Get real-time recommendations
    real_time_recs = recommender.get_real_time_recommendations(user_id, n_recommendations=5)
    
    # If user has no preferences yet, return popular songs
    if not real_time_recs:
        real_time_recs = recommender.get_popular_songs(n_songs=5)
    
    return jsonify({
        'success': True, 
        'recommendations': real_time_recs
    })

# Helper functions
def get_user_liked_songs(user_id):
    """Get the list of songs liked by the user"""
    # Initialize liked songs as empty list
    liked_songs = []
    
    # Check if we have a session key for liked songs
    session_key = f'user_{user_id}_likes'
    if session_key in session:
        return session[session_key]
    
    # Otherwise, check recommender object
    if recommender is not None and user_id in recommender.user_preferences:
        for song_id in recommender.user_preferences[user_id]:
            if song_id in recommender.reverse_mapping:
                song_key = recommender.reverse_mapping[song_id]
                # Find song in data
                song_df = recommender.song_data[recommender.song_data['song_id'] == song_id]
                if not song_df.empty:
                    song = song_df.iloc[0]
                    liked_songs.append({
                        'artist': song['artist'],
                        'song': song['song'],
                        'song_key': song_key,
                        'link': song['link']
                    })
    
    # Cache in session
    session[session_key] = liked_songs
    return liked_songs

def is_song_liked(user_id, song_key):
    """Check if a song is liked by the user"""
    if recommender is None:
        return False
        
    liked_songs = get_user_liked_songs(user_id)
    return any(song['song_key'] == song_key for song in liked_songs)

def get_more_from_artist(artist_name, exclude_song=None, n_songs=4):
    """Get more songs by the same artist"""
    songs = []
    
    if recommender is None:
        return []
    
    artist_songs = recommender.song_data[recommender.song_data['artist'] == artist_name]
    
    if exclude_song:
        # Filter out the current song
        artist_songs = artist_songs[~artist_songs.apply(lambda row: f"{row['artist']} - {row['song']}" == exclude_song, axis=1)]
    
    # Sample songs if we have more than requested
    if len(artist_songs) > n_songs:
        artist_songs = artist_songs.sample(n_songs)
    
    for _, song in artist_songs.iterrows():
        songs.append({
            'artist': song['artist'],
            'song': song['song'],
            'song_key': f"{song['artist']} - {song['song']}",
            'link': song['link']
        })
    
    return songs

def get_trending_songs(n_songs=5):
    """Get trending songs (random for demo)."""
    trending_songs = []
    
    if recommender:
        sample_songs = recommender.song_data.sample(n_songs)
        
        for _, song in sample_songs.iterrows():
            trending_songs.append({
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link'],
                'popularity': round(random.uniform(70, 95))  # Random popularity score for demo
            })
    
    return trending_songs

def get_new_releases(n_songs=5):
    """Get new releases (random for demo)."""
    new_releases = []
    
    if recommender:
        sample_songs = recommender.song_data.sample(n_songs)
        
        for _, song in sample_songs.iterrows():
            new_releases.append({
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link'],
                'release_date': f"202{random.randint(0, 3)}-{random.randint(1, 12):02d}"  # Random recent date for demo
            })
    
    return new_releases

# Helper function to generate dummy recommendations when model is not available
def get_dummy_recommendations(n_songs=8):
    dummy_songs = [
        {'artist': 'The Beatles', 'song': 'Hey Jude', 'song_key': 'The Beatles - Hey Jude', 'score': 0.9, 'reason': 'Popular song', 'is_liked': False},
        {'artist': 'Queen', 'song': 'Bohemian Rhapsody', 'song_key': 'Queen - Bohemian Rhapsody', 'score': 0.88, 'reason': 'Popular song', 'is_liked': False},
        {'artist': 'Michael Jackson', 'song': 'Billie Jean', 'song_key': 'Michael Jackson - Billie Jean', 'score': 0.85, 'reason': 'Popular song', 'is_liked': False},
        {'artist': 'Led Zeppelin', 'song': 'Stairway to Heaven', 'song_key': 'Led Zeppelin - Stairway to Heaven', 'score': 0.83, 'reason': 'Classic rock', 'is_liked': False},
        {'artist': 'Pink Floyd', 'song': 'Comfortably Numb', 'song_key': 'Pink Floyd - Comfortably Numb', 'score': 0.81, 'reason': 'Classic rock', 'is_liked': False},
        {'artist': 'The Rolling Stones', 'song': 'Paint It Black', 'song_key': 'The Rolling Stones - Paint It Black', 'score': 0.79, 'reason': 'Classic rock', 'is_liked': False},
        {'artist': 'Eagles', 'song': 'Hotel California', 'song_key': 'Eagles - Hotel California', 'score': 0.77, 'reason': 'Classic rock', 'is_liked': False},
        {'artist': 'Nirvana', 'song': 'Smells Like Teen Spirit', 'song_key': 'Nirvana - Smells Like Teen Spirit', 'score': 0.75, 'reason': '90s rock', 'is_liked': False}
    ]
    return dummy_songs[:n_songs]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003) 