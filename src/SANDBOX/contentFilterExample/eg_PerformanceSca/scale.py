from corerec.engines.content_based.performance_scalability.scalable_algorithms import ScalableAlgorithms
from corerec.engines.content_based.other_approaches import (
    OTH_RULE_BASED,
    OTH_SENTIMENT_ANALYSIS,
    OTH_ONTOLOGY_BASED
)

def filter_content_item(item, rule_filter, sentiment_filter, ontology_filter):
    """
    Applies all filters to a single content item.

    Parameters:
    - item (str): The content to filter.
    - rule_filter (RuleBasedFilter): An instance of RuleBasedFilter.
    - sentiment_filter (SentimentAnalysisFilter): An instance of SentimentAnalysisFilter.
    - ontology_filter (OntologyBasedFilter): An instance of OntologyBasedFilter.

    Returns:
    - dict: Aggregated filtering results.
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
    rule_filter = OTH_RULE_BASED()
    rule_filter.add_rule("sci-fi", "flag")
    rule_filter.add_rule("fantastic", "allow")

    sentiment_filter = OTH_SENTIMENT_ANALYSIS(threshold=0.2)

    ontology_filter = OTH_ONTOLOGY_BASED("src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl")

    # Initialize Scalable Algorithms
    scalable = ScalableAlgorithms()

    # Define a wrapper function for parallel processing
    def process_item(item):
        return filter_content_item(item, rule_filter, sentiment_filter, ontology_filter)

    # Use parallel_process to filter content concurrently
    filtered_results = scalable.parallel_process(process_item, content_data)

    # Display Results
    for idx, res in enumerate(filtered_results, start=1):
        print(f"Content Item {idx}:")
        print(res)
        print("-" * 40)

if __name__ == "__main__":
    main()