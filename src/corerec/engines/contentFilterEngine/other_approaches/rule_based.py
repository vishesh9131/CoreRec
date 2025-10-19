# rule_based implementation
class RuleBasedFilter:
    def __init__(self, rules=None):
        """
        Initializes the RuleBasedFilter with a set of rules.

        Parameters:
        - rules (list of dict): A list where each rule is a dictionary containing
                                 'keyword' and 'action' keys.
        """
        if rules is None:
            self.rules = []
        else:
            self.rules = rules

    def add_rule(self, keyword, action):
        """
        Adds a new rule to the filter.

        Parameters:
        - keyword (str): The keyword to look for in the content.
        - action (str): The action to take ('block', 'flag', etc.).
        """
        rule = {'keyword': keyword.lower(), 'action': action.lower()}
        self.rules.append(rule)

    def filter_content(self, content):
        """
        Filters the content based on the predefined rules.

        Parameters:
        - content (str): The content to be filtered.

        Returns:
        - dict: A dictionary with 'status' and 'actions' applied.
        """
        actions_applied = []
        content_lower = content.lower()

        for rule in self.rules:
            if rule['keyword'] in content_lower:
                actions_applied.append(rule['action'])

        if 'block' in actions_applied:
            return {'status': 'blocked', 'actions': actions_applied}
        elif 'flag' in actions_applied:
            return {'status': 'flagged', 'actions': actions_applied}
        else:
            return {'status': 'allowed', 'actions': actions_applied}
