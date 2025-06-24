@echo off
REM QuantConnect Development Helper (Windows Batch)

set VENV_PYTHON=.venv\Scripts\python.exe

if "%1"=="test" goto test
if "%1"=="install" goto install
if "%1"=="init" goto init
if "%1"=="backtest" goto backtest
if "%1"=="debug" goto debug
goto help

:help
echo QuantConnect Development Commands:
echo   qc.bat test        - Test environment setup
echo   qc.bat install     - Install required packages  
echo   qc.bat init        - Initialize LEAN project
echo   qc.bat backtest    - Run backtest
echo   qc.bat debug       - Run algorithm in debug mode
goto end

:test
echo Testing environment...
%VENV_PYTHON% test_environment.py
goto end

:install
echo Installing packages...
%VENV_PYTHON% -m pip install --upgrade pip
%VENV_PYTHON% -m pip install -r requirements.txt
%VENV_PYTHON% -m pip install lean
goto end

:init
echo Initializing LEAN project...
%VENV_PYTHON% -m lean init
goto end

:backtest
echo Running backtest...
%VENV_PYTHON% -m lean backtest QuantConnect --verbose
goto end

:debug
echo Starting algorithm in debug mode...
echo Use VS Code debugger instead (F5) for better debugging experience
%VENV_PYTHON% quantconnect\main.py
goto end

:end
