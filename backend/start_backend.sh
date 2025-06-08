#!/bin/bash

echo "Starting Resume Refiner AI Backend..."

# Activate virtual environment
source venv/Scripts/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please run setup_backend.sh first to set up the environment."
    exit 1
fi

# Start the Flask server
echo "Starting Flask server..."
python app.py 