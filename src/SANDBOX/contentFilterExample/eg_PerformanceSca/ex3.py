import logging
from corerec.engines.content_based.other_approaches import (
    OTH_RULE_BASED,
    OTH_SENTIMENT_ANALYSIS,
    OTH_ONTOLOGY_BASED
)
from corerec.engines.content_based.performance_scalability import (
    PER_SCALABLE_ALGORITHMS,
    PER_FEATURE_EXTRACTION,
    PER_LOAD_BALANCING
)
from corerec.engines.content_based.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_USER_PROFILING,
    CON_ITEM_PROFILING
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_content_item_all(
    item: str,
    rule_filter: OTH_RULE_BASED,
    sentiment_filter: OTH_SENTIMENT_ANALYSIS,
    ontology_filter: OTH_ONTOLOGY_BASED,
    feature_extractor: PER_FEATURE_EXTRACTION,
    context_recommender: CON_CONTEXT_AWARE,
    user_recommender: CON_USER_PROFILING,
    item_recommender: CON_ITEM_PROFILING
) -> dict:
    """
    Applies all filters to a single content item, extracts features, and generates personalized recommendations.

    Parameters:
    - item (str): The content to filter.
    - rule_filter (RuleBasedFilter): An instance of RuleBasedFilter.
    - sentiment_filter (SentimentAnalysisFilter): An instance of SentimentAnalysisFilter.
    - ontology_filter (OntologyBasedFilter): An instance of OntologyBasedFilter.
    - feature_extractor (FeatureExtraction): An instance of FeatureExtraction.
    - context_recommender (ContextAwareRecommender): An instance of ContextAwareRecommender.
    - user_recommender (UserProfilingRecommender): An instance of UserProfilingRecommender.
    - item_recommender (ItemProfilingRecommender): An instance of ItemProfilingRecommender.

    Returns:
    - dict: Aggregated results including filtering statuses, feature vectors, and recommendations.
    """
    result = {}
    # Apply Rule-Based Filter
    rule_result = rule_filter.filter_content(item)
    result['rule_based'] = rule_result

    # Apply Sentiment Analysis Filter
    sentiment_result = sentiment_filter.filter_content(item)
    result['sentiment_analysis'] = sentiment_result

    # Apply Ontology-Based Filter
    ontology_result = ontology_filter.filter_content(item)
    result['ontology_based'] = ontology_result

    # Extract Features
    feature_vector = feature_extractor.transform([item])
    result['features'] = feature_vector

    # Generate Context-Aware Recommendations (Assuming item_id is known)
    # For demonstration, we'll assign a dummy item_id
    dummy_item_id = 1  # Replace with actual logic to determine item_id
    context_recommendation = context_recommender.recommend(dummy_item_id)
    result['context_recommendation'] = context_recommendation

    # Define Item Features
    item_features = {
        1: {"genre": "SciFi", "chemistry": "High"},
        2: {"genre": "Disaster", "plot": "Poor"},
        3: {"genre": "Documentary", "topics": "Various"},
        4: {"genre": "Action", "thriller": "High"},
        5: {"genre": "RomanticDrama", "feelings": "Indifferent"}
    }

    # Extract item IDs from item_features
    item_ids = list(item_features.keys())

    all_items = set(item_ids)
    
    # Generate User Profiling Recommendations
    user_id = 1  # Replace with actual user context
    user_recommendation = user_recommender.recommend(user_id, all_items, top_n=10)
    result['user_recommendation'] = user_recommendation

    # Generate Item Profiling Recommendations
    item_profiling_recommendation = item_recommender.recommend(dummy_item_id)
    result['item_profiling_recommendation'] = item_profiling_recommendation

    return result

def main():
    # Sample Content Data
    content_data = [
        "This is a SciFi movie about space exploration.",
        "An action-packed disaster film with stunning visuals.",
        "A documentary on the wonders of chemistry.",
        "An exciting action thriller with a unique plot twist.",
        "A romantic drama that explores deep emotional connections."
    ]

    # Initialize Other Approaches Filters
    rule_filter = OTH_RULE_BASED()
    rule_filter.add_rule("SciFi", action="flag")
    rule_filter.add_rule("action", action="allow")
    rule_filter.add_rule("disaster", action="block")
    rule_filter.add_rule("documentary", action="allow")
    rule_filter.add_rule("romantic", action="neutral")

    sentiment_filter = OTH_SENTIMENT_ANALYSIS(threshold=0.1)
    ontology_filter = OTH_ONTOLOGY_BASED("src/SANDBOX/contentFilterExample/otheraoprach_eg/ontologies/ontology.owl")

    # Initialize Performance Scalability Components
    feature_extractor = PER_FEATURE_EXTRACTION(max_features=1000)
    feature_extractor.fit_transform(content_data)  # Fit on initial data

    # Path to Context Configuration File
    context_config_path = "src/SANDBOX/contentFilterExample/eg_PerformanceSca/context_config.json"  # Update the path accordingly

    # Define Item Features
    item_features = {
        1: {"genre": "SciFi", "chemistry": "High"},
        2: {"genre": "Disaster", "plot": "Poor"},
        3: {"genre": "Documentary", "topics": "Various"},
        4: {"genre": "Action", "thriller": "High"},
        5: {"genre": "RomanticDrama", "feelings": "Indifferent"}
    }

    # Initialize Context Personalization Components
    context_recommender = CON_CONTEXT_AWARE(context_config_path, item_features)
    user_recommender = CON_USER_PROFILING()
    item_recommender = CON_ITEM_PROFILING()

    # Initialize Load Balancer
    load_balancer = PER_LOAD_BALANCING(num_workers=4)

    # Add tasks to the LoadBalancer
    for item in content_data:
        load_balancer.add_task(
            filter_content_item_all,
            item,
            rule_filter,
            sentiment_filter,
            ontology_filter,
            feature_extractor,
            context_recommender,
            user_recommender,
            item_recommender
        )

    # Retrieve and display results
    filtered_results = load_balancer.get_results()
    load_balancer.shutdown()

    for idx, res in enumerate(filtered_results, start=1):
        print(f"Content Item {idx}:")
        print("Rule-Based Filter:", res['rule_based'])
        print("Sentiment Analysis Filter:", res['sentiment_analysis'])
        print("Ontology-Based Filter:", res['ontology_based'])
        print("Feature Vector (TF-IDF):\n", res['features'].toarray())
        print("Context-Aware Recommendation:", res['context_recommendation'])
        print("User Profiling Recommendation:", res['user_recommendation'])
        print("Item Profiling Recommendation:", res['item_profiling_recommendation'])
        print("-" * 40)

    user_interactions = {
        1: [1, 2, 3],
        2: [2, 3, 4],
        3: [1, 4, 5]
    }

    # Train the recommenders
    context_recommender.fit(user_interactions)
    item_recommender.fit(user_interactions, item_features)

if __name__ == "__main__":
    main() 