# Import using cleaner module path
import corerec as cr
import os


def main():
    content = (
        "I absolutely love the new sci-fi movie! The chemistry between characters is fantastic."
    )

    # Rule-Based Filtering
    rule_filter = cr.RuleBased()
    rule_filter.add_rule("sci-fi", "flag")
    rule_filter.add_rule("fantastic", "allow")
    rule_result = rule_filter.filter_content(content)
    print("Rule-Based Filter Result:", rule_result)

    # Sentiment Analysis Filtering
    sentiment_filter = cr.SentimentAnalysis(threshold=0.2)
    sentiment_result = sentiment_filter.filter_content(content)
    print("Sentiment Analysis Filter Result:", sentiment_result)

    # Ontology-Based Filtering
    # Try different possible paths for the ontology file
    possible_paths = [
        "src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl",
        "examples/ContentFilterExamples/eg_otherApproach/ontologies/ontology.owl",
        "../ontologies/ontology.owl",
        "ontologies/ontology.owl",
    ]
    
    ontology_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ontology_path = path
            break
    
    if ontology_path:
        try:
            ontology_filter = cr.OntologyBased(ontology_path)
            ontology_result = ontology_filter.filter_content(content)
            print("Ontology-Based Filter Result:", ontology_result)
        except Exception as e:
            print(f"Ontology-Based Filter Error: {e}")
            print("Skipping ontology-based filtering.")
    else:
        print("Ontology file not found. Skipping ontology-based filtering.")
        print("Tried paths:", possible_paths)


if __name__ == "__main__":
    main()
