#!/bin/bash

echo "Starting Resume Refiner AI Backend Setup..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed!"
    echo "Please install Python 3.10 or higher and try again."
    exit 1
fi

# Remove existing virtual environment if it exists
rm -rf venv

# Create new virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages with specific versions
echo "Installing required packages..."

# Core packages
pip install flask==3.0.2
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.1
pip install werkzeug==3.0.1

# Data science packages - carefully tested combination
pip install numpy==1.26.4
pip install pandas==2.2.1
pip install scikit-learn==1.4.1

# ML and AI packages - versions that work with the above
pip install torch==2.2.1
pip install huggingface-hub==0.21.4
pip install sentence-transformers==2.5.1
pip install transformers==4.38.2
pip install joblib==1.3.2

# Document processing packages
pip install PyPDF2==3.0.1
pip install python-docx==1.1.0

# Utility packages
pip install requests==2.31.0
pip install tqdm==4.66.2
pip install colorama==0.4.6

# Google AI package
pip install google-generativeai==0.3.2

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p models
mkdir -p cache

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    echo "Please update the .env file with your Google API key."
fi

echo "Setup completed successfully!"
echo "To start the server, run: ./start_backend.sh" 