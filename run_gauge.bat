@echo off
REM Launch the GAUGE desktop/web application on Windows.
REM Usage: double-click this file, or run it from an Anaconda Prompt with the
REM "gauge" conda environment active (see INSTALL instructions in README.md).

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if "%GAUGE_PORT%"=="" set GAUGE_PORT=8501

python -m streamlit run app\Home.py --server.port %GAUGE_PORT% --server.headless true
pause
