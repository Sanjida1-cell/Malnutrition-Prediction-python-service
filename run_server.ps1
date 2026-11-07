# Malnutrition Prediction Multi-Model API PowerShell Start Script
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting Malnutrition Prediction Multi-Model API" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Locate Python executable
$pythonCmd = (Get-Command python, py -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $pythonCmd) {
    $defaultPy = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python313\python.exe"
    if (Test-Path $defaultPy) {
        $pythonCmd = $defaultPy
    } else {
        Write-Error "Python 3 is not found. Please install Python 3.10+ and add to PATH."
        exit 1
    }
}

# Create virtual environment if missing
if (-not (Test-Path "venv")) {
    Write-Host "[INFO] Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv venv
}

# Activate virtual environment
$activateScript = "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Yellow
    & $activateScript
}

# Install dependencies
Write-Host "[INFO] Installing/verifying dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Verify model files in models folder
if (-not (Test-Path "models")) {
    Write-Error "The models/ directory was not found!"
    exit 1
}
$modelFiles = Get-ChildItem -Path "models" -Filter "*.pkl"
if ($modelFiles.Count -eq 0) {
    Write-Error "No .pkl model files found in models/ directory!"
    exit 1
}
Write-Host "[INFO] Found $($modelFiles.Count) trained model files in models/ folder." -ForegroundColor Green

# Start Uvicorn server
Write-Host "[INFO] Starting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "[INFO] Models endpoint: http://127.0.0.1:8000/models" -ForegroundColor Green
Write-Host "[INFO] Interactive docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
