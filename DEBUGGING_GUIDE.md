# QuantConnect Local Development Setup Guide

## 🎯 What You Now Have Set Up

### 1. VS Code Configuration Files
- **`.vscode/launch.json`** - Debug configurations for your RL algorithm
- **`.vscode/tasks.json`** - Build and run tasks 
- **`.vscode/settings.json`** - Python environment and QuantConnect settings

### 2. Helper Scripts
- **`qc.bat`** - Windows batch script for common commands
- **`qc.ps1`** - PowerShell script with more features
- **`test_environment.py`** - Environment verification script
- **`debug_local.py`** - Local debugging helper

### 3. Configuration Files
- **`lean.json`** - LEAN engine configuration

## 🚀 How to Use for Debugging

### Method 1: VS Code Debugging (Recommended)
1. Open your algorithm file (`quantconnect/main.py`)
2. Set breakpoints by clicking in the left margin
3. Press **F5** or go to **Run and Debug** panel
4. Select "Debug QuantConnect Algorithm"
5. Your code will stop at breakpoints for inspection

### Method 2: VS Code Tasks
1. Press **Ctrl+Shift+P**
2. Type "Tasks: Run Task"
3. Choose from:
   - LEAN: Backtest Algorithm
   - LEAN: Initialize Project  
   - Install Required Packages

### Method 3: Terminal Commands
```bash
# Test your setup
python test_environment.py

# Run local debugging
python debug_local.py

# Install packages
pip install -r requirements.txt
```

## 🔧 Fixing Virtual Environment Issues

If you see Python path errors, try:

```powershell
# Reactivate virtual environment
.\.venv\Scripts\Activate.ps1

# Or create a new one
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🐛 Debugging Your RL Algorithm

### Key Debugging Points:
1. **Model Loading** - Set breakpoints in `ModelWrapper.LoadModel()`
2. **Observation Generation** - Debug `get_observation()` method
3. **Action Execution** - Check `execute_option_trades()`
4. **Market Data** - Verify `OnData()` processing

### Common Issues to Debug:
- Model weights loading correctly
- Observation vector shape and values
- Action scaling and execution
- Option chain filtering
- Portfolio state tracking

## 📊 Using QuantConnect Extension

The QuantConnect extension provides:
- Cloud integration
- Real-time debugging
- Backtest management
- Research environment

Configure it in `.vscode/settings.json` with your QuantConnect credentials.

## 🔍 Next Steps

1. **Test the environment**: Run `python test_environment.py`
2. **Set breakpoints** in your main algorithm
3. **Use F5 debugging** to step through your code
4. **Monitor variables** in the VS Code debugger
5. **Check logs** in the integrated terminal

Your RL hedging algorithm debugging setup is now complete! 🎉
