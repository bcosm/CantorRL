#!/usr/bin/env python3
"""
Test script to verify QuantConnect/LEAN setup
"""
import sys
import os

def test_environment():
    print("=== QuantConnect Development Environment Test ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Test package imports
    packages_to_test = [
        'numpy',
        'pandas', 
        'torch',
        'pickle',
        'io',
        'typing'
    ]
    
    print("\n=== Package Tests ===")
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✓ {package} - OK")
        except ImportError as e:
            print(f"✗ {package} - FAILED: {e}")
    
    # Test LEAN CLI
    print("\n=== LEAN CLI Test ===")
    try:
        import lean
        print("✓ LEAN CLI - Installed")
        
        # Try to get LEAN version
        try:
            from lean.commands.version import version_command
            print("✓ LEAN CLI - Commands available")
        except ImportError:
            print("⚠ LEAN CLI - Commands might not be available")
            
    except ImportError as e:
        print(f"✗ LEAN CLI - Not installed: {e}")
    
    # Test file structure
    print("\n=== File Structure Test ===")
    files_to_check = [
        'quantconnect/main.py',
        'quantconnect/model_wrapper.py',
        'quantconnect/option_calculator.py',
        'requirements.txt',
        '.vscode/launch.json',
        '.vscode/tasks.json',
        'lean.json'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} - Exists")
        else:
            print(f"✗ {file_path} - Missing")
    
    print("\n=== Environment Setup Complete ===")
    print("You can now use:")
    print("1. VS Code debugger with F5")
    print("2. VS Code tasks (Ctrl+Shift+P -> Tasks: Run Task)")
    print("3. Terminal commands for LEAN")

if __name__ == "__main__":
    test_environment()
