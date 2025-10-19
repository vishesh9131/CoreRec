from textblob import TextBlob

class SentimentAnalysisFilter:
    def __init__(self, threshold=0.1):
        """
        Initializes the SentimentAnalysisFilter.

        Parameters:
        - threshold (float): The sentiment polarity threshold to trigger actions.
                             Positive values can indicate positive sentiment,
                             negative values indicate negative sentiment.
        """
        self.threshold = threshold

    def analyze_sentiment(self, content):
        """
        Analyzes the sentiment of the given content.

        Parameters:
        - content (str): The content to analyze.

        Returns:
        - float: The sentiment polarity score ranging from -1.0 to 1.0.
        """
        blob = TextBlob(content)
        return blob.sentiment.polarity

    def filter_content(self, content):
        """
        Filters the content based on its sentiment.

        Parameters:
        - content (str): The content to be filtered.

        Returns:
        - dict: A dictionary with 'status' and 'sentiment_score'.
        """
        sentiment_score = self.analyze_sentiment(content)

        if sentiment_score < -self.threshold:
            return {'status': 'negative', 'sentiment_score': sentiment_score}
        elif sentiment_score > self.threshold:
            return {'status': 'positive', 'sentiment_score': sentiment_score}
        else:
            return {'status': 'neutral', 'sentiment_score': sentiment_score}
