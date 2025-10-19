/**
 * Main JavaScript for CoreRec Demo Frontends Platform Selector
 */

class CoreRecDemoApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.userId = null;
        this.platforms = [];
        
        this.init();
    }

    async init() {
        await this.loadPlatforms();
        this.setupEventListeners();
        await this.createUser();
    }

    async loadPlatforms() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/platforms`);
            const data = await response.json();
            this.platforms = data.platforms;
        } catch (error) {
            console.error('Failed to load platforms:', error);
            this.showError('Failed to connect to API. Please ensure the backend is running.');
        }
    }

    async createUser() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/users/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            this.userId = data.user_id;
            console.log('Created user session:', this.userId);
        } catch (error) {
            console.error('Failed to create user:', error);
        }
    }

    setupEventListeners() {
        // Platform launch buttons
        const launchButtons = document.querySelectorAll('.launch-btn');
        launchButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const platform = button.getAttribute('data-platform');
                this.launchPlatform(platform);
            });
        });

        // Card click handlers
        const platformCards = document.querySelectorAll('.platform-card');
        platformCards.forEach(card => {
            card.addEventListener('click', (e) => {
                // Only trigger if not clicking the button
                if (!e.target.closest('.launch-btn')) {
                    const platform = card.getAttribute('data-platform');
                    this.launchPlatform(platform);
                }
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideLoading();
            }
        });
    }

    async launchPlatform(platformId) {
        const platform = this.platforms.find(p => p.id === platformId);
        if (!platform) {
            this.showError('Platform not found');
            return;
        }

        this.showLoading(platform.name);

        try {
            // Simulate platform setup
            await this.setupPlatform(platformId);
            
            // Navigate to platform
            setTimeout(() => {
                window.location.href = `platforms/${platformId}/index.html?user_id=${this.userId}`;
            }, 1500);

        } catch (error) {
            console.error('Failed to launch platform:', error);
            this.hideLoading();
            this.showError(`Failed to launch ${platform.name}. Please try again.`);
        }
    }

    async setupPlatform(platformId) {
        // Load platform data to ensure it's ready
        const response = await fetch(`${this.apiBaseUrl}/platforms/${platformId}/data?limit=10`);
        if (!response.ok) {
            throw new Error(`Failed to load ${platformId} data`);
        }
        
        const data = await response.json();
        console.log(`Loaded ${data.total_items} items for ${platformId}`);
        
        return data;
    }

    showLoading(platformName) {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        const loadingDetails = document.getElementById('loading-details');
        
        loadingText.textContent = `Launching ${platformName}...`;
        loadingDetails.textContent = 'Setting up your demo environment';
        
        overlay.classList.remove('hidden');
        
        // Update loading text progressively
        setTimeout(() => {
            loadingDetails.textContent = 'Loading recommendation data...';
        }, 500);
        
        setTimeout(() => {
            loadingDetails.textContent = 'Preparing user interface...';
        }, 1000);
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.add('hidden');
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-message">${message}</span>
                <button class="error-close">&times;</button>
            </div>
        `;
        
        // Add styles
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4444;
            color: white;
            padding: 16px 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(255, 68, 68, 0.3);
            z-index: 2000;
            max-width: 400px;
            animation: slideIn 0.3s ease-out;
        `;
        
        // Add to DOM
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
        
        // Close button handler
        const closeBtn = errorDiv.querySelector('.error-close');
        closeBtn.addEventListener('click', () => {
            errorDiv.remove();
        });
        
        // Add animation styles if not already present
        if (!document.querySelector('#error-animation-styles')) {
            const styles = document.createElement('style');
            styles.id = 'error-animation-styles';
            styles.textContent = `
                @keyframes slideIn {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                .error-content {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                .error-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.2rem;
                    cursor: pointer;
                    padding: 0;
                    margin-left: auto;
                }
                .error-close:hover {
                    opacity: 0.8;
                }
            `;
            document.head.appendChild(styles);
        }
    }

    // Utility method to check API status
    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/`);
            const data = await response.json();
            return data.status === 'running';
        } catch (error) {
            return false;
        }
    }
}

// Animation utilities
class AnimationUtils {
    static observeElements() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, {
            threshold: 0.1
        });

        // Observe platform cards
        document.querySelectorAll('.platform-card').forEach(card => {
            observer.observe(card);
        });

        // Observe feature cards
        document.querySelectorAll('.feature').forEach(feature => {
            observer.observe(feature);
        });
    }

    static addScrollEffects() {
        let ticking = false;

        function updateScrollEffects() {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            
            const header = document.querySelector('.header-bg');
            if (header) {
                header.style.transform = `translateY(${rate}px)`;
            }
            
            ticking = false;
        }

        function requestTick() {
            if (!ticking) {
                requestAnimationFrame(updateScrollEffects);
                ticking = true;
            }
        }

        window.addEventListener('scroll', requestTick);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize main app
    const app = new CoreRecDemoApp();
    
    // Initialize animations
    AnimationUtils.observeElements();
    AnimationUtils.addScrollEffects();
    
    // Add dynamic effects
    const platformCards = document.querySelectorAll('.platform-card');
    platformCards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in-up');
    });
    
    // Console welcome message
    console.log(`
    🚀 CoreRec Demo Frontends
    ========================
    Welcome to the CoreRec demonstration platform!
    
    This application showcases recommendation models through
    beautiful, platform-specific interfaces.
    
    Available platforms: Spotify, YouTube, Netflix
    `);
});

// Add CSS animations
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    .fade-in-up {
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.6s ease-out forwards;
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-in {
        animation: bounceIn 0.6s ease-out;
    }

    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.8) translateY(20px);
        }
        60% {
            opacity: 1;
            transform: scale(1.05) translateY(-5px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
`;
document.head.appendChild(animationStyles); 