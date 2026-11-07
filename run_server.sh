#!/bin/bash

# Malnutrition Prediction API Server Start Script
# Starts the FastAPI server for the malnutrition prediction service (Multi-Model Set C)

echo "🚀 Starting Malnutrition Prediction Multi-Model API Server..."
echo "=================================="

# Check if Python is available (try python3, then python)
if command -v python3 &> /dev/null; then
    PY_CMD="python3"
elif command -v python &> /dev/null; then
    PY_CMD="python"
else
    echo "❌ Error: Python is not installed or not in PATH"
    echo "Please install Python 3.10+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PY_CMD -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install dependencies
echo "📋 Installing/updating dependencies..."
pip install -r requirements.txt

# Check for model files in models directory
MODEL_COUNT=$(ls -1 models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ Error: No .pkl model files found in models/ directory"
    exit 1
fi
echo "✅ Found $MODEL_COUNT trained model files in models/ directory"

# Start the server
echo "🌐 Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "Available Models endpoint: http://localhost:8000/models"
echo "Interactive API Documentation: http://localhost:8000/docs"
echo "Alternative Documentation: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
