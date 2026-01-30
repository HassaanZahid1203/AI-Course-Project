@echo off
echo Installing Python packages for Python 3.14.2...

REM ========================================
REM STEP 1 — Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM ========================================
REM STEP 2 — Install core libraries
echo Installing numpy, pillow, pyqt5, opencv-python...
pip install numpy pillow pyqt5 opencv-python

REM ========================================
REM STEP 3 — Install PyTorch (CPU version)
echo Installing PyTorch (CPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM ========================================
REM STEP 4 — Install dlib
echo Installing dlib...

REM Attempt to install via pip (may fail if no prebuilt wheel exists)
pip install dlib

IF ERRORLEVEL 1 (
    echo.
    echo ----------------------------------------
    echo dlib failed to install via pip.
    echo You will likely need to install from a wheel.
    echo Please download the appropriate dlib wheel for Python 3.14.2 manually,
    echo from a trusted source such as the unofficial Gohlke repository:
    echo https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib
    echo Then run:
    echo pip install path\to\dlib‑*.whl
    echo ----------------------------------------
)

echo.
echo Installation complete!
pause
