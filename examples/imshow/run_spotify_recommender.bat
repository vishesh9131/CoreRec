@echo off
:: Run Spotify Recommender script for Windows
:: This batch script helps run the Spotify Recommender from any directory

:: Get the directory of this script (project root)
set "SCRIPT_DIR=%~dp0"

:: Path to the Spotify Recommender directory
set "RECOMMENDER_DIR=%SCRIPT_DIR%examples\CollaborativeFilterExamples\spotify_recommender"

:: Check if the directory exists
if not exist "%RECOMMENDER_DIR%" (
    echo Error: Spotify Recommender directory not found at %RECOMMENDER_DIR%
    exit /b 1
)

:: Change to the Spotify Recommender directory
cd /d "%RECOMMENDER_DIR%"
echo Changed to directory: %RECOMMENDER_DIR%

:: Run the application
echo Starting Spotify Recommender...
python app.py

:: Exit with the same status as the Python script
exit /b %ERRORLEVEL% 