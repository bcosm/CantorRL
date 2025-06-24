"""
Quick component test for QuantConnect algorithm without full framework
"""

def test_model_wrapper_structure():
    """Test model wrapper file structure"""
    try:
        with open('quantconnect/model_wrapper.py', 'r') as f:
            content = f.read()
          print("✅ ModelWrapper file found")
        print("File size: {} characters".format(len(content)))
        
        # Check for key components
        key_methods = ['LoadModel', 'predict', 'LoadSB3Weights']
        for method in key_methods:
            if method in content:
                print("✓ Found method: {}".format(method))
            else:
                print("⚠ Missing method: {}".format(method))
                
    except Exception as e:
        print(f"❌ Error reading model_wrapper.py: {e}")

def test_main_algorithm_structure():
    """Test main algorithm structure"""
    try:
        with open('quantconnect/main.py', 'r') as f:
            content = f.read()
          print("Main algorithm file found")
        print("File size: {} characters".format(len(content)))
        
        # Check for key components
        key_components = [
            'class RLHedgingAlgorithm',
            'def initialize',
            'def OnData',
            'ModelWrapper',
            'get_observation',
            'execute_option_trades'
        ]
        
        for component in key_components:            if component in content:
                print("Found: {}".format(component))
            else:
                print("Missing: {}".format(component))
                
    except Exception as e:
        print(f"❌ Error reading main.py: {e}")

def main():
    print("🧪 QuantConnect Algorithm Component Test")
    print("=" * 50)
    
    test_model_wrapper_structure()
    test_main_algorithm_structure()
    
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATION:")
    print("Use VS Code debugging (F5) for the best development experience!")
    print("This will let you:")
    print("• Set breakpoints")
    print("• Inspect variables") 
    print("• Step through your RL model logic")
    print("• Debug option trading decisions")

if __name__ == "__main__":
    main()
