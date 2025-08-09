// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners to all like buttons
    const likeButtons = document.querySelectorAll('.like-btn');
    
    likeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const songKey = this.getAttribute('data-song-key');
            const isLiked = this.classList.contains('liked');
            
            // Toggle liked state
            if (isLiked) {
                unlikeSong(songKey, this);
            } else {
                likeSong(songKey, this);
            }
        });
    });
    
    // Function to like a song
    function likeSong(songKey, button) {
        fetch('/api/like_song', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ song_key: songKey }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                button.classList.add('liked');
                button.innerHTML = '<i class="fas fa-heart"></i>';
                
                // Show real-time recommendations if available
                if (data.recommendations && data.recommendations.length > 0) {
                    showRealTimeRecommendations(data.recommendations, songKey);
                }
            }
        })
        .catch(error => {
            console.error('Error liking song:', error);
        });
    }
    
    // Function to unlike a song
    function unlikeSong(songKey, button) {
        fetch('/api/unlike_song', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ song_key: songKey }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI
                button.classList.remove('liked');
                button.innerHTML = '<i class="far fa-heart"></i>';
                
                // Update recommendations if available
                if (data.recommendations && data.recommendations.length > 0) {
                    updateRecommendationsSection(data.recommendations);
                }
            }
        })
        .catch(error => {
            console.error('Error unliking song:', error);
        });
    }
    
    // Function to show real-time recommendations in a notification
    function showRealTimeRecommendations(recommendations, likedSongKey) {
        // Create or get the real-time recommendations container
        let realtimeContainer = document.getElementById('realtime-recommendations');
        
        if (!realtimeContainer) {
            realtimeContainer = document.createElement('div');
            realtimeContainer.id = 'realtime-recommendations';
            realtimeContainer.className = 'realtime-recommendations';
            document.body.appendChild(realtimeContainer);
            
            // Add styles if not already in CSS
            if (!document.getElementById('realtime-recommendation-styles')) {
                const styles = document.createElement('style');
                styles.id = 'realtime-recommendation-styles';
                styles.innerHTML = `
                    .realtime-recommendations {
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        background-color: #181818;
                        border-radius: 8px;
                        padding: 16px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
                        width: 320px;
                        max-height: 80vh;
                        overflow-y: auto;
                        z-index: 1000;
                        transition: transform 0.3s ease-in-out;
                        transform: translateX(100%);
                    }
                    .realtime-recommendations.show {
                        transform: translateX(0);
                    }
                    .realtime-recommendations h3 {
                        margin-top: 0;
                        margin-bottom: 16px;
                        font-size: 18px;
                        display: flex;
                        justify-content: space-between;
                    }
                    .realtime-recommendations .close-btn {
                        background: none;
                        border: none;
                        color: #b3b3b3;
                        cursor: pointer;
                        font-size: 16px;
                    }
                    .realtime-recommendations .close-btn:hover {
                        color: #ffffff;
                    }
                    .realtime-recommendations .rec-list {
                        list-style: none;
                        padding: 0;
                        margin: 0;
                    }
                    .realtime-recommendations .rec-item {
                        padding: 8px 0;
                        border-bottom: 1px solid #333;
                        display: flex;
                        align-items: center;
                    }
                    .realtime-recommendations .rec-item:last-child {
                        border-bottom: none;
                    }
                    .realtime-recommendations .rec-icon {
                        width: 50px;
                        height: 50px;
                        margin-right: 12px;
                        border-radius: 4px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background-color: #282828;
                        color: #1DB954;
                        font-size: 24px;
                    }
                    .realtime-recommendations .rec-info {
                        flex-grow: 1;
                    }
                    .realtime-recommendations .rec-song {
                        font-size: 14px;
                        font-weight: 600;
                        margin: 0;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .realtime-recommendations .rec-artist {
                        font-size: 12px;
                        color: #b3b3b3;
                        margin: 4px 0 0;
                    }
                    .realtime-recommendations .rec-reason {
                        font-size: 11px;
                        color: #1DB954;
                        margin: 4px 0 0;
                    }
                `;
                document.head.appendChild(styles);
            }
        }
        
        // Get the liked song details
        const likedSongParts = likedSongKey.split(' - ');
        const likedArtist = likedSongParts[0];
        const likedSong = likedSongParts[1];
        
        // Create recommendation content
        let content = `
            <h3>
                Because you liked "${likedSong}"
                <button class="close-btn"><i class="fas fa-times"></i></button>
            </h3>
            <ul class="rec-list">
        `;
        
        recommendations.forEach(rec => {
            content += `
                <li class="rec-item">
                    <div class="rec-icon">
                        <i class="fas fa-music"></i>
                    </div>
                    <div class="rec-info">
                        <p class="rec-song">
                            <a href="/song/${rec.song_key.replace('&', '_and_')}">${rec.song}</a>
                        </p>
                        <p class="rec-artist">
                            <a href="/artist/${rec.artist.replace('&', '_and_')}">${rec.artist}</a>
                        </p>
                        <p class="rec-reason">${rec.reason}</p>
                    </div>
                </li>
            `;
        });
        
        content += `</ul>`;
        
        // Update the container and show it
        realtimeContainer.innerHTML = content;
        
        // Add close button functionality
        const closeButton = realtimeContainer.querySelector('.close-btn');
        closeButton.addEventListener('click', function() {
            realtimeContainer.classList.remove('show');
        });
        
        // Show the recommendations with a slight delay
        setTimeout(() => {
            realtimeContainer.classList.add('show');
        }, 100);
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            realtimeContainer.classList.remove('show');
        }, 10000);
    }
    
    // Function to update the recommendations section
    function updateRecommendationsSection(recommendations) {
        const recommendationsSection = document.querySelector('.recommendations-section .text-cards');
        if (!recommendationsSection) return;
        
        // Clear existing recommendations
        recommendationsSection.innerHTML = '';
        
        // Add new recommendations
        recommendations.forEach(rec => {
            const songCard = document.createElement('div');
            songCard.className = 'text-card';
            
            songCard.innerHTML = `
                <div class="text-icon">
                    <i class="fas fa-music"></i>
                </div>
                <div class="song-info">
                    <h3><a href="/song/${rec.song_key.replace('&', '_and_')}">${rec.song}</a></h3>
                    <p><a href="/artist/${rec.artist.replace('&', '_and_')}">${rec.artist}</a></p>
                    <div class="song-meta">
                        <span class="confidence">Score: ${rec.score.toFixed(2)}</span>
                        <span class="reason">${rec.reason}</span>
                    </div>
                    <div class="song-actions">
                        <button class="like-btn" data-song-key="${rec.song_key}">
                            <i class="far fa-heart"></i>
                        </button>
                    </div>
                </div>
            `;
            
            // Add to recommendations section
            recommendationsSection.appendChild(songCard);
            
            // Add event listener to the new like button
            const newLikeButton = songCard.querySelector('.like-btn');
            newLikeButton.addEventListener('click', function() {
                const songKey = this.getAttribute('data-song-key');
                const isLiked = this.classList.contains('liked');
                
                if (isLiked) {
                    unlikeSong(songKey, this);
                } else {
                    likeSong(songKey, this);
                }
            });
        });
    }
}); 