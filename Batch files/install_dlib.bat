@echo off
echo Installing dlib for Python 3.14.2...

REM Upgrade pip first (optional but recommended)
python -m pip install --upgrade pip

REM Install dlib from precompiled wheel
pip install dlib-bin

echo.
echo dlib installation complete!
pause
