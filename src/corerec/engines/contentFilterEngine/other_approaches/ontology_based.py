from owlready2 import get_ontology

class OntologyBasedFilter:
    def __init__(self, ontology_path):
        """
        Initializes the OntologyBasedFilter with a specific ontology.

        Parameters:
        - ontology_path (str): The file path to the ontology (.owl) file.
        """
        try:
            self.ontology = get_ontology(ontology_path).load()
        except Exception as e:
            raise ValueError(f"Failed to load ontology from {ontology_path}: {e}")

    def get_concepts(self, content):
        """
        Extracts concepts from the content based on the ontology.

        Parameters:
        - content (str): The content to extract concepts from.

        Returns:
        - set: A set of concepts identified in the content.
        """
        concepts_found = set()
        content_lower = content.lower()

        for cls in self.ontology.classes():
            if cls.name.lower() in content_lower:
                concepts_found.add(cls.name)

        return concepts_found

    def filter_content(self, content):
        """
        Filters the content based on ontology-defined relationships.

        Parameters:
        - content (str): The content to be filtered.

        Returns:
        - dict: A dictionary with 'status' and 'related_concepts'.
        """
        concepts = self.get_concepts(content)
        related_concepts = self.find_related_concepts(concepts)

        if related_concepts:
            return {'status': 'filtered', 'related_concepts': related_concepts}
        else:
            return {'status': 'allowed', 'related_concepts': related_concepts}

    def find_related_concepts(self, concepts):
        """
        Finds related concepts within the ontology.

        Parameters:
        - concepts (set): A set of concepts to find relationships for.

        Returns:
        - dict: A dictionary mapping each concept to its related concepts.
        """
        related = {}
        for concept in concepts:
            try:
                cls = self.ontology[concept]
                related[concept] = [str(rel) for rel in cls.is_a]
            except KeyError:
                related[concept] = []
        return related
