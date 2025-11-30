import logging
import corerec as cr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_content_item_all(
    item: str,
    rule_filter: cr.RuleBased,
    sentiment_filter: cr.SentimentAnalysis,
    ontology_filter: cr.OntologyBased,
    feature_extractor: cr.FeatureExtraction,
    context_recommender: cr.ContextAware,
    user_recommender: cr.UserProfiling,
    item_recommender: cr.ItemProfiling,
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
    result["rule_based"] = rule_result

    # Apply Sentiment Analysis Filter
    sentiment_result = sentiment_filter.filter_content(item)
    result["sentiment_analysis"] = sentiment_result

    # Apply Ontology-Based Filter
    if ontology_filter:
        ontology_result = ontology_filter.filter_content(item)
        result["ontology_based"] = ontology_result
    else:
        result["ontology_based"] = {"status": "skipped", "reason": "ontology file not found"}

    # Extract Features
    feature_vector = feature_extractor.transform([item])
    result["features"] = feature_vector

    # Define Item Features
    item_features = {
        1: {"genre": "SciFi", "chemistry": "High"},
        2: {"genre": "Disaster", "plot": "Poor"},
        3: {"genre": "Documentary", "topics": "Various"},
        4: {"genre": "Action", "thriller": "High"},
        5: {"genre": "RomanticDrama", "feelings": "Indifferent"},
    }

    # Extract item IDs from item_features
    item_ids = list(item_features.keys())
    all_items = set(item_ids)

    # Generate Context-Aware Recommendations (Assuming user_id is known)
    # For demonstration, we'll assign a dummy user_id
    dummy_user_id = 1  # Replace with actual logic to determine user_id
    try:
        context_recommendation = context_recommender.recommend(dummy_user_id)
        result["context_recommendation"] = context_recommendation
    except Exception as e:
        logger.warning(f"Context recommendation failed: {e}")
        result["context_recommendation"] = []

    # Generate User Profiling Recommendations
    user_id = 1  # Replace with actual user context
    try:
        user_recommendation = user_recommender.recommend(user_id, all_items, top_n=10)
        result["user_recommendation"] = user_recommendation
    except Exception as e:
        logger.warning(f"User recommendation failed: {e}")
        result["user_recommendation"] = []

    # Generate Item Profiling Recommendations
    dummy_item_id = 1  # For demonstration
    try:
        item_profiling_recommendation = item_recommender.recommend(dummy_item_id)
        result["item_profiling_recommendation"] = item_profiling_recommendation
    except Exception as e:
        logger.warning(f"Item profiling recommendation failed: {e}")
        result["item_profiling_recommendation"] = []

    return result


def main():
    # Sample Content Data
    content_data = [
        "This is a SciFi movie about space exploration.",
        "An action-packed disaster film with stunning visuals.",
        "A documentary on the wonders of chemistry.",
        "An exciting action thriller with a unique plot twist.",
        "A romantic drama that explores deep emotional connections.",
    ]

    # Initialize Other Approaches Filters
    rule_filter = cr.RuleBased()
    rule_filter.add_rule("SciFi", action="flag")
    rule_filter.add_rule("action", action="allow")
    rule_filter.add_rule("disaster", action="block")
    rule_filter.add_rule("documentary", action="allow")
    rule_filter.add_rule("romantic", action="neutral")

    sentiment_filter = cr.SentimentAnalysis(threshold=0.1)
    
    # Try different possible paths for the ontology file
    import os
    possible_ontology_paths = [
        "src/SANDBOX/contentFilterExample/otheraoprach_eg/ontologies/ontology.owl",
        "src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl",
        "examples/ContentFilterExamples/eg_otherApproach/ontologies/ontology.owl",
    ]
    
    ontology_path = None
    for path in possible_ontology_paths:
        if os.path.exists(path):
            ontology_path = path
            break
    
    if ontology_path:
        try:
            ontology_filter = cr.OntologyBased(ontology_path)
        except Exception as e:
            logger.warning(f"Ontology file not found, skipping ontology filtering: {e}")
            ontology_filter = None
    else:
        logger.warning("Ontology file not found. Skipping ontology-based filtering.")
        ontology_filter = None

    # Initialize Performance Scalability Components
    feature_extractor = cr.FeatureExtraction(max_features=1000)
    feature_extractor.fit_transform(content_data)  # Fit on initial data

    # Path to Context Configuration File
    context_config_path = "src/SANDBOX/contentFilterExample/eg_PerformanceSca/context_config.json"  # Update the path accordingly

    # Define Item Features
    item_features = {
        1: {"genre": "SciFi", "chemistry": "High"},
        2: {"genre": "Disaster", "plot": "Poor"},
        3: {"genre": "Documentary", "topics": "Various"},
        4: {"genre": "Action", "thriller": "High"},
        5: {"genre": "RomanticDrama", "feelings": "Indifferent"},
    }

    # Initialize Context Personalization Components
    context_recommender = cr.ContextAware(context_config_path, item_features)
    user_recommender = cr.UserProfiling()
    item_recommender = cr.ItemProfiling()

    # Train the recommenders BEFORE using them
    user_interactions = {1: [1, 2, 3], 2: [2, 3, 4], 3: [1, 4, 5]}
    context_recommender.fit(user_interactions)
    user_recommender.fit(user_interactions)
    item_recommender.fit(user_interactions, item_features)

    # Initialize Load Balancer
    load_balancer = cr.LoadBalancing(num_workers=4)

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
            item_recommender,
        )

    # Retrieve and display results
    filtered_results = load_balancer.get_results()
    load_balancer.shutdown()

    for idx, res in enumerate(filtered_results, start=1):
        print(f"Content Item {idx}:")
        print("Rule-Based Filter:", res["rule_based"])
        print("Sentiment Analysis Filter:", res["sentiment_analysis"])
        print("Ontology-Based Filter:", res["ontology_based"])
        if res["features"] is not None:
            print("Feature Vector (TF-IDF):\n", res["features"].toarray())
        else:
            print("Feature Vector (TF-IDF): [Not available]")
        print("Context-Aware Recommendation:", res["context_recommendation"])
        print("User Profiling Recommendation:", res["user_recommendation"])
        print("Item Profiling Recommendation:", res["item_profiling_recommendation"])
        print("-" * 40)

    # Recommenders are already fitted above before load balancing


if __name__ == "__main__":
    main()
