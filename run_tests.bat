@echo off
echo Checking Python version...
python --version

echo.
echo Checking Python path...
python -c "import sys; print('\n'.join(sys.path))"

echo.
echo Checking pip list...
python -m pip list

echo.
echo Running simple test...
python -c "print('Hello from Python!')"

echo.
echo Running pytest...
python -m pytest tests/test_minimal.py -v
