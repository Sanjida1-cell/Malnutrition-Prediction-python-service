#!/bin/bash

# Malnutrition Prediction API Server Start Script
# This script starts the FastAPI server for the malnutrition prediction service

echo "🚀 Starting Malnutrition Prediction API Server..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "ensemble_model.pkl" ]; then
    echo "❌ Error: Model file 'ensemble_model.pkl' not found"
    echo "Please ensure the model file is in the current directory"
    exit 1
fi

# Start the server
echo "🌐 Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Alternative docs: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
