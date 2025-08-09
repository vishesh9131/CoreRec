# CoreShop: DLRM Recommendation Demo Website

This project is a demo e-commerce website that showcases the Deep Learning Recommendation Model (DLRM) from the CoreRec framework. The website demonstrates how personalized recommendations can enhance the shopping experience.

## Features

- **DLRM-Powered Recommendations**: View personalized product recommendations based on the DLRM model
- **User Switching**: Switch between different user profiles to see how recommendations change
- **Product Browsing**: Browse products by category
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive UI**: Modern and user-friendly interface
- **Recommendation Explanation**: Learn about how the DLRM model generates recommendations

## Requirements

- Python 3.6+
- Flask
- NumPy (version < 2.0)
- PyTorch
- Pandas

## Installation

1. Make sure you have trained the DLRM model first by running the example:
```bash
python examples/CollaborativeFilterExamples/dlrm_eg.py
```

2. Install the required dependencies:
```bash
pip install Flask numpy<2.0 torch pandas
```

3. Navigate to the website directory:
```bash
cd examples/CollaborativeFilterExamples/dlrm_website
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and go to:
```
http://localhost:5000
```

## Usage

- **Home Page**: View featured products and personalized recommendations
- **Product Page**: View product details and related products
- **Category Page**: Browse products by category
- **User Switching**: Click on "Switch User" in the top-right corner to change user profiles and see different recommendations

## How It Works

1. The website loads the pre-trained DLRM model from `dlrm_ijcai_model.pt`
2. User information and activity data from the IJCAI-15 dataset is used to generate features
3. The DLRM model generates personalized recommendations based on these features
4. The recommendations are displayed on the website with confidence scores

## Demo Data

The website uses:
- Real user data from the IJCAI-15 dataset
- Real merchant IDs from the dataset
- Synthetic product information for demonstration purposes

## Directory Structure

```
dlrm_website/
├── app.py                # Flask application
├── static/               # Static files
│   ├── css/              # CSS styles
│   ├── js/               # JavaScript files
│   └── img/              # Images
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── product.html      # Product detail page
│   └── category.html     # Category page
└── README.md             # This file
```

## Customization

- Add your own product images to `static/img/`
- Modify the CSS styles in `static/css/style.css`
- Extend the JavaScript functionality in `static/js/main.js`

## Credits

- The DLRM model is based on the paper "Deep Learning Recommendation Model for Personalization and Recommendation Systems" by Naumov et al.
- The CoreRec framework provides the DLRM implementation
- The IJCAI-15 dataset is used for demonstration purposes

## License

This project is for demonstration purposes only.

## Screenshots

![Home Page](screenshots/home.jpg)
![Product Page](screenshots/product.jpg)
![Category Page](screenshots/category.jpg) 