# Recsys Inspired by Youtube DNN Architecture


**I have broken down the Youtube DNN architecture into several parts**

1. **Taking Care of Data & Basic Recsys Setup**
    - Setup and Import
    - Data Prep
    - Initialize models from engine
    - Fit the models
    - Initialize Hybrid Engine
    - Use the hybrid engine to generate recommendations for a specific user.

2. **Implementing a Mixture of Experts (MoE)**
    - creating a custom MoE layer (BaseRecommender)
    - Integrate MoE into Hybrid Engine
    - Fit the models
    - Generate Recommendations
3. **Handling Position Bias**
    - Extend Recommender with position "Features" (PositionAwareMFRecommender)
    - Integrate Position-Aware Recommender
4. **Evaluation**
    - Offline Metrics
      - AUC (Area Under Curve) for classification tasks like click-through prediction.
      - Mean Squared Error (MSE) for regression tasks like watch time prediction.
    - Online Metrics
      - Engagement Metrics: Clicks, watch time.
      - Satisfaction Metrics: Likes, ratings.

## Conclusion
By following the steps outlined above, you can create a robust recommendation system clone using the corerec library. The modular design of corerec allows you to experiment with various algorithms and ensemble methods, enabling you to tailor the system to meet specific objectives like user engagement and satisfaction.
Feel free to extend the functionality by integrating more sophisticated models, handling biases, and optimizing performance based on your requirements.