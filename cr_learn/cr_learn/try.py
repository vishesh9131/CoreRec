import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cr_learn.rees46 import load, preprocess_events

def load_and_preprocess(sample_size: float = 0.01, use_columns: list = None, nrows: int = 1000) -> pd.DataFrame:
    """Load and preprocess the data with options for sampling, column selection, and row limit."""
    data = load(sample_size=sample_size, use_columns=use_columns)
    events_df = data['events']
    
    if nrows:
        events_df = events_df.head(nrows)
    
    return preprocess_events(events_df)

def visualize_data(df: pd.DataFrame):
    """Visualize the distribution of event prices."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Distribution of Event Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

events_df = load_and_preprocess(sample_size=0.01, use_columns=['event_time', 'price', 'brand'], nrows=1000)

visualize_data(events_df)
