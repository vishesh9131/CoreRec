/**
 * Main JavaScript for CoreShop DLRM Demo
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all interactive elements
    initializeAddToCartButtons();
    initializeTooltips();
    highlightRecommendationScores();
    initializeSearchBar();
});

/**
 * Initialize Add to Cart buttons
 */
function initializeAddToCartButtons() {
    const addToCartButtons = document.querySelectorAll('.add-to-cart');
    
    addToCartButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get product info
            const productCard = this.closest('.product-card');
            let productName = 'this item';
            
            if (productCard) {
                const productNameElement = productCard.querySelector('h3 a');
                if (productNameElement) {
                    productName = productNameElement.textContent;
                }
            }
            
            // Show a notification
            showNotification(`Added ${productName} to cart!`, 'success');
            
            // Animate the cart icon
            animateCartIcon();
        });
    });
}

/**
 * Show a notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    // Add to the DOM
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Close button functionality
    const closeButton = notification.querySelector('.notification-close');
    closeButton.addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    });
    
    // Auto close after 3 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.classList.remove('show');
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    notification.remove();
                }
            }, 300);
        }
    }, 3000);
}

/**
 * Animate the cart icon
 */
function animateCartIcon() {
    const cartIcon = document.querySelector('.cart i');
    if (cartIcon) {
        cartIcon.classList.add('animate-cart');
        setTimeout(() => {
            cartIcon.classList.remove('animate-cart');
        }, 500);
    }
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    // Custom tooltips are handled via CSS
    // This function can be expanded for more complex tooltip behavior
}

/**
 * Highlight recommendation scores
 */
function highlightRecommendationScores() {
    const recScores = document.querySelectorAll('.rec-score');
    
    recScores.forEach(score => {
        const scoreText = score.textContent.trim();
        const scoreValue = parseInt(scoreText.match(/\d+/)[0], 10);
        
        if (scoreValue >= 90) {
            score.classList.add('high-score');
        } else if (scoreValue >= 70) {
            score.classList.add('medium-score');
        } else {
            score.classList.add('low-score');
        }
    });
}

/**
 * Initialize search bar
 */
function initializeSearchBar() {
    const searchBar = document.querySelector('.search-bar input');
    const searchButton = document.querySelector('.search-bar button');
    
    if (searchBar && searchButton) {
        searchButton.addEventListener('click', function() {
            handleSearch(searchBar.value);
        });
        
        searchBar.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleSearch(searchBar.value);
            }
        });
    }
}

/**
 * Handle search functionality
 */
function handleSearch(query) {
    if (query.trim() === '') {
        showNotification('Please enter a search term', 'info');
        return;
    }
    
    showNotification(`Search functionality is not implemented in this demo`, 'info');
}

/**
 * Add custom animation for notification
 */
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: white;
        border-radius: 4px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        padding: 15px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        z-index: 1000;
        transform: translateX(120%);
        transition: transform 0.3s ease;
        max-width: 350px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-content {
        display: flex;
        align-items: center;
    }
    
    .notification-content i {
        margin-right: 10px;
        font-size: 1.2em;
    }
    
    .notification.success .notification-content i {
        color: #2ecc71;
    }
    
    .notification.info .notification-content i {
        color: #3498db;
    }
    
    .notification-close {
        background: none;
        border: none;
        cursor: pointer;
        margin-left: 15px;
        color: #999;
    }
    
    .notification-close:hover {
        color: #333;
    }
    
    @keyframes cartBounce {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }
    
    .animate-cart {
        animation: cartBounce 0.5s;
        color: #4a90e2;
    }
    
    .rec-score.high-score {
        background-color: rgba(46, 204, 113, 0.9);
    }
    
    .rec-score.medium-score {
        background-color: rgba(243, 156, 18, 0.9);
    }
    
    .rec-score.low-score {
        background-color: rgba(231, 76, 60, 0.9);
    }
`;

document.head.appendChild(style); 