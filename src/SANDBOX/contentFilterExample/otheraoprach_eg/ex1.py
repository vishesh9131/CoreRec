from corerec.engines.contentFilterEngine.other_approaches import (
    OTH_RULE_BASED,
    OTH_SENTIMENT_ANALYSIS,
    OTH_ONTOLOGY_BASED
)

def main():
    content = "I absolutely love the new sci-fi movie! The chemistry between characters is fantastic."

    # Rule-Based Filtering
    rule_filter = OTH_RULE_BASED()
    rule_filter.add_rule("sci-fi", "flag")
    rule_filter.add_rule("fantastic", "allow")
    rule_result = rule_filter.filter_content(content)
    print("Rule-Based Filter Result:", rule_result)

    # Sentiment Analysis Filtering
    sentiment_filter = OTH_SENTIMENT_ANALYSIS(threshold=0.2)
    sentiment_result = sentiment_filter.filter_content(content)
    print("Sentiment Analysis Filter Result:", sentiment_result)

    # Ontology-Based Filtering
    ontology_filter = OTH_ONTOLOGY_BASED("src/SANDBOX/contentFilterExample/exampleotheraoprach/ontologies/ontology.owl")
    ontology_result = ontology_filter.filter_content(content)
    print("Ontology-Based Filter Result:", ontology_result)

if __name__ == "__main__":
    main()
