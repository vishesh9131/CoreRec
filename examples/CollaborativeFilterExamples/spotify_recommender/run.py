#!/usr/bin/env python
"""
Run script for Spotify Music Recommender
This script ensures the application runs from the correct directory
"""
import os
import sys
from pathlib import Path

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    print(f"Changed working directory to: {script_dir}")
    
    # Run the app
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5003)
    except Exception as e:
        print(f"Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 