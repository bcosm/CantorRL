# QuantConnect Development Helper Script
# This script provides easy commands for working with QuantConnect/LEAN

param(
    [string]$Command = "help"
)

$VenvPath = ".\.venv\Scripts"
$PythonExe = "$VenvPath\python.exe"

function Show-Help {
    Write-Host "QuantConnect Development Commands:" -ForegroundColor Green
    Write-Host "  .\qc.ps1 test        - Test environment setup"
    Write-Host "  .\qc.ps1 install     - Install required packages"
    Write-Host "  .\qc.ps1 init        - Initialize LEAN project"
    Write-Host "  .\qc.ps1 backtest    - Run backtest"
    Write-Host "  .\qc.ps1 research    - Start research environment"
    Write-Host "  .\qc.ps1 debug       - Run algorithm in debug mode"
}

function Test-Environment {
    Write-Host "Testing environment..." -ForegroundColor Yellow
    & $PythonExe test_environment.py
}

function Install-Packages {
    Write-Host "Installing packages..." -ForegroundColor Yellow
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt
    & $PythonExe -m pip install lean
}

function Initialize-Lean {
    Write-Host "Initializing LEAN project..." -ForegroundColor Yellow
    & $PythonExe -m lean init
}

function Run-Backtest {
    Write-Host "Running backtest..." -ForegroundColor Yellow
    & $PythonExe -m lean backtest "QuantConnect" --verbose
}

function Start-Research {
    Write-Host "Starting research environment..." -ForegroundColor Yellow
    & $PythonExe -m lean research
}

function Debug-Algorithm {
    Write-Host "Starting algorithm in debug mode..." -ForegroundColor Yellow
    Write-Host "Use VS Code debugger instead (F5) for better debugging experience" -ForegroundColor Cyan
    & $PythonExe quantconnect\main.py
}

switch ($Command.ToLower()) {
    "help" { Show-Help }
    "test" { Test-Environment }
    "install" { Install-Packages }
    "init" { Initialize-Lean }
    "backtest" { Run-Backtest }
    "research" { Start-Research }
    "debug" { Debug-Algorithm }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help 
    }
}
