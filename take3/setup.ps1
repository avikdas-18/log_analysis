$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
python -m venv venv

Write-Host "Activating virtual environment..."
$env:Path = "$pwd\venv\Scripts;$env:Path"
$env:VIRTUAL_ENV = "$pwd\venv"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
# Installing streamlit, pandas, transformers, torch, sentence-transformers
pip install streamlit pandas transformers torch sentence-transformers

Write-Host "Setup complete!"
