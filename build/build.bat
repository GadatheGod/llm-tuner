@echo off
echo Building LLM-Tuner with PyInstaller...
echo.

pyinstaller --onedir --name "LLM-Tuner" ^
    --add-data "llmtuner/data;llmtuner/data" ^
    --hidden-import PySide6.QtWidgets ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtGui ^
    --hidden-import psutil ^
    --hidden-import GPUtil ^
    --hidden-import cpuinfo ^
    --hidden-import requests ^
    main.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful! Run: dist\LLM-Tuner\LLM-Tuner.exe
    echo.
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    echo.
)

pause
