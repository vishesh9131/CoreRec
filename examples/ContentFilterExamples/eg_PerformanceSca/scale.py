import corerec as cr
from functools import partial


def filter_content_item(item, rule_filter, sentiment_filter, ontology_filter):
    """
    Applies all filters to a single content item.

    Parameters:
    - item (str): The content to filter.
    - rule_filter (RuleBasedFilter): An instance of RuleBasedFilter.
    - sentiment_filter (SentimentAnalysisFilter): An instance of SentimentAnalysisFilter.
    - ontology_filter (OntologyBasedFilter): An instance of OntologyBasedFilter or None.

    Returns:
    - dict: Aggregated filtering results.
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

    return result


def main():
    # Sample data
    content_data = [
        "I absolutely love the new sci-fi movie! The chemistry between characters is fantastic.",
        "This is a terrible disaster movie with no plot.",
        "An average documentary that covers various topics.",
        # Add more content items as needed
    ]

    # Initialize Filters
    rule_filter = cr.RuleBased()
    rule_filter.add_rule("sci-fi", "flag")
    rule_filter.add_rule("fantastic", "allow")

    sentiment_filter = cr.SentimentAnalysis(threshold=0.2)

    # Try different possible paths for the ontology file
    import os
    possible_ontology_paths = [
        "src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl",
        "src/SANDBOX/contentFilterExample/otheraoprach_eg/ontologies/ontology.owl",
        "examples/ContentFilterExamples/eg_otherApproach/ontologies/ontology.owl",
    ]
    
    ontology_path = None
    for path in possible_ontology_paths:
        if os.path.exists(path):
            ontology_path = path
            break
    
    ontology_filter = None
    if ontology_path:
        try:
            ontology_filter = cr.OntologyBased(ontology_path)
        except Exception as e:
            print(f"Warning: Could not load ontology file: {e}")
    else:
        print("Warning: Ontology file not found. Skipping ontology-based filtering.")

    # Initialize Scalable Algorithms
    scalable = cr.ScalableAlgorithms()

    # Create a partial function with filters bound
    process_item = partial(filter_content_item, 
                          rule_filter=rule_filter, 
                          sentiment_filter=sentiment_filter,
                          ontology_filter=ontology_filter)

    # Use parallel_process to filter content concurrently
    filtered_results = scalable.parallel_process(process_item, content_data)

    # Display Results
    for idx, res in enumerate(filtered_results, start=1):
        print(f"Content Item {idx}:")
        print(res)
        print("-" * 40)


if __name__ == "__main__":
    main()
