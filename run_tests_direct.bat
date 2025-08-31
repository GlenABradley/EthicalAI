@echo off
set PYTHON_PATH=C:\Users\glen4\AppData\Local\Programs\Python\Python312\python.exe

echo Checking Python version...
%PYTHON_PATH% --version

echo.
echo Checking Python path...
%PYTHON_PATH% -c "import sys; print('\n'.join(sys.path))"

echo.
echo Checking pip list...
%PYTHON_PATH% -m pip list

echo.
echo Running simple test...
%PYTHON_PATH% -c "print('Hello from Python!')"

echo.
echo Running pytest...
%PYTHON_PATH% -m pytest tests/test_minimal.py -v
