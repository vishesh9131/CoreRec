# Core Concepts

This guide introduces the fundamental concepts behind CoreRec and how recommendation systems work.

## What is CoreRec?

CoreRec is a comprehensive framework for building recommendation systems that combines:

- **Deep Learning Models**: Neural network-based approaches for capturing complex patterns
- **Collaborative Filtering**: User-item interaction-based recommendations
- **Content-Based Filtering**: Feature-based recommendations
- **Hybrid Methods**: Combining multiple approaches for better results

## Core Concepts

### Recommendation Systems

A recommendation system suggests items to users based on:
- **User preferences**: Past interactions, ratings, behavior
- **Item features**: Content, metadata, attributes
- **Context**: Time, location, device, etc.

### Model Types

#### 1. Collaborative Filtering (CF)
Predicts user preferences by finding similar users or items.

**User-Based CF**: "Users similar to you also liked..."
**Item-Based CF**: "Users who liked this also liked..."

#### 2. Content-Based Filtering
Recommends items similar to what the user has liked before, based on item features.

#### 3. Deep Learning Models
Neural networks that learn complex patterns from user-item interactions.

#### 4. Hybrid Methods
Combine multiple approaches to leverage strengths of each.

### Key Components

#### Base Recommender
All models inherit from `BaseRecommender`, providing:
- Unified API (`fit`, `predict`, `recommend`)
- Model persistence (`save`, `load`)
- Consistent error handling

#### Engines
CoreRec organizes models into engines:
- **Deep Learning Engine**: Neural network models
- **Unionized Filter Engine**: Collaborative filtering
- **Content Filter Engine**: Content-based methods

### Workflow

1. **Data Preparation**: Load and preprocess your dataset
2. **Model Selection**: Choose appropriate model for your use case
3. **Training**: Fit the model on your data
4. **Evaluation**: Measure performance metrics
5. **Deployment**: Use model for recommendations

## Next Steps

- Read the [Quickstart Guide](quickstart.md) to get started
- Explore [Model Documentation](models/index.md) for specific models
- Check out [Examples](examples/basic_usage.md) for code samples

