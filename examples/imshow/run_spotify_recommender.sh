#!/bin/bash

# Run Spotify Recommender script
# This shell script helps run the Spotify Recommender from any directory

# Get the directory of this script (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the Spotify Recommender directory
RECOMMENDER_DIR="$SCRIPT_DIR/examples/CollaborativeFilterExamples/spotify_recommender"

# Check if the directory exists
if [ ! -d "$RECOMMENDER_DIR" ]; then
    echo "Error: Spotify Recommender directory not found at $RECOMMENDER_DIR"
    exit 1
fi

# Change to the Spotify Recommender directory
cd "$RECOMMENDER_DIR"
echo "Changed to directory: $RECOMMENDER_DIR"

# Run the application
echo "Starting Spotify Recommender..."
python app.py

# Exit with the same status as the Python script
exit $? 