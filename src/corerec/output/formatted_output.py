# output/formatted_output.py
import pandas as pd
import json
from typing import List, Optional

class OutputFormatter:
    """
    Formats recommendation outputs for easy interpretation and further usage.
    """

    def format_recommendations(
        self,
        recommendations: List[int],
        item_metadata: Optional[pd.DataFrame] = None,
        user_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Converts a list of recommended item IDs into a structured DataFrame.
        
        Parameters:
        - recommendations (List[int]): List of recommended item IDs.
        - item_metadata (Optional[pd.DataFrame]): DataFrame containing item details.
        - user_id (Optional[int]): The ID of the user for whom recommendations are generated.
        
        Returns:
        - pd.DataFrame: Formatted recommendations with optional item metadata.
        """
        try:
            rec_df = pd.DataFrame(recommendations, columns=['ItemID'])
            if user_id is not None:
                rec_df.insert(0, 'UserID', user_id)
            
            if item_metadata is not None and 'ItemID' in item_metadata.columns:
                rec_df = rec_df.merge(item_metadata, on='ItemID', how='left')
            
            print("Formatted recommendations into DataFrame successfully.")
            return rec_df
        except Exception as e:
            print(f"Error formatting recommendations: {e}")
            raise

    def export_recommendations(
        self,
        rec_df: pd.DataFrame,
        file_path: str,
        file_format: str = 'csv'
    ) -> None:
        """
        Exports the formatted recommendations to a specified file format.
        
        Parameters:
        - rec_df (pd.DataFrame): The formatted recommendations DataFrame.
        - file_path (str): The destination file path.
        - file_format (str): The format to export the data ('csv' or 'json').
        """
        try:
            if file_format.lower() == 'csv':
                rec_df.to_csv(file_path, index=False)
                print(f"Exported recommendations to CSV at {file_path}.")
            elif file_format.lower() == 'json':
                rec_df.to_json(file_path, orient='records', lines=True)
                print(f"Exported recommendations to JSON at {file_path}.")
            else:
                raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
        except Exception as e:
            print(f"Error exporting recommendations: {e}")
            raise