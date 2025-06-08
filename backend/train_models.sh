#!/bin/bash

echo "Starting Resume Refiner AI Model Training..."

# Activate virtual environment
source venv/Scripts/activate

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: data directory not found!"
    echo "Please create a 'data' directory and add the required datasets:"
    echo "1. IT_Job_Roles_Skills.csv"
    echo "2. UpdatedResumeDataSet.csv"
    exit 1
fi

# Check if required datasets exist
if [ ! -f "data/IT_Job_Roles_Skills.csv" ] || [ ! -f "data/UpdatedResumeDataSet.csv" ]; then
    echo "Error: Required datasets not found!"
    echo "Please add the following files to the 'data' directory:"
    echo "1. IT_Job_Roles_Skills.csv"
    echo "2. UpdatedResumeDataSet.csv"
    exit 1
fi

# Run the training script
echo "Running model training..."
python train_models.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Model training completed successfully!"
    echo "Models have been saved to the 'models' directory."
else
    echo "Error: Model training failed!"
    exit 1
fi

pause 