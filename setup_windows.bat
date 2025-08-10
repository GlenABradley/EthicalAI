@echo off
REM Setup script for Windows

echo Setting up Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements-windows.txt

echo Creating .env file...
echo # Environment variables > .env
echo COHERENCE_DEVICE=auto >> .env
echo COHERENCE_SEED=42 >> .env

echo Setup complete! Activate the environment with:
echo .venv\Scripts\activate.bat
