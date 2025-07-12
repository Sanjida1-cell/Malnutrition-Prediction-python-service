#!/bin/bash

# Step 1: Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Step 2: Activate the virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Start the FastAPI server
echo "Starting FastAPI server at http://127.0.0.1:8000 ..."
uvicorn main:app --reload
