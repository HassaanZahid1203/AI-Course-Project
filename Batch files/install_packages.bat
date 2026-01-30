@echo off
echo Installing Python packages for Python 3.14.2...

REM Upgrade pip first (optional but recommended)
python -m pip install --upgrade pip

REM Install dlib from precompiled wheel
echo Installing dlib...
pip install dlib-bin

REM Install Seaborn
echo Installing seaborn...
pip install seaborn

REM Install Matplotlib
echo Installing matplotlib...
pip install matplotlib

echo.
echo All installations complete!
pause
