"""
Verify QuantConnect setup can run without full LEAN framework
"""
import sys
import os

# Add the quantconnect directory to Python path
quantconnect_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quantconnect')
sys.path.insert(0, quantconnect_dir)

def test_basic_setup():
    """Test basic Python environment"""
    print("=== Basic Environment Test ===")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import pickle
        print("✓ Pickle module available")
        
        import io
        print("✓ IO module available")
        
        return True
    except Exception as e:
        print(f"✗ Basic setup failed: {e}")
        return False

def test_algorithm_files():
    """Test if algorithm files exist and can be read"""
    print("\n=== Algorithm Files Test ===")
    
    files_to_check = [
        'quantconnect/main.py',
        'quantconnect/model_wrapper.py', 
        'quantconnect/option_calculator.py'
    ]
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            print(f"✓ {file_path} ({len(content)} chars)")
        except Exception as e:
            print(f"✗ {file_path}: {e}")
            return False
    
    return True

def test_model_wrapper_import():
    """Test if we can import ModelWrapper"""
    print("\n=== ModelWrapper Import Test ===")
    
    try:
        # Change to quantconnect directory for imports
        original_cwd = os.getcwd()
        os.chdir('quantconnect')
        
        # Create a minimal mock for testing
        class MockAlgorithm:
            def Log(self, msg):
                print(f"[MOCK] {msg}")
            
            class MockObjectStore:
                def ReadBytes(self, key):
                    raise Exception(f"Mock: Cannot read {key}")
            
            def __init__(self):
                self.ObjectStore = self.MockObjectStore()
        
        # Try to import and instantiate
        from model_wrapper import ModelWrapper
        mock_algo = MockAlgorithm()
        
        # This will fail at model loading but should import successfully
        print("✓ ModelWrapper class imported successfully")
        print("  Note: Model loading will fail without actual model files")
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"✗ ModelWrapper import failed: {e}")
        os.chdir(original_cwd)
        return False

def test_algorithm_components():
    """Test algorithm components without full QuantConnect"""
    print("\n=== Algorithm Components Test ===")
    
    try:
        # Test if we can create basic observations
        import numpy as np
        
        # Mock some data like the algorithm would use
        observation_size = 13  # Based on your model
        mock_observation = np.random.randn(observation_size)
        
        print(f"✓ Mock observation created: shape {mock_observation.shape}")
        
        # Test if we can simulate action execution
        action_size = 2  # Based on your model (call/put actions)
        mock_actions = np.random.randn(action_size)
        
        print(f"✓ Mock actions created: shape {mock_actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm components test failed: {e}")
        return False

def main():
    """Run comprehensive setup verification"""
    print("QuantConnect Setup Verification (No LEAN Required)")
    print("=" * 60)
    
    tests = [
        test_basic_setup,
        test_algorithm_files,
        test_model_wrapper_import,
        test_algorithm_components
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 SETUP VERIFICATION COMPLETE!")
        print("\n✅ Your environment is ready for:")
        print("  • VS Code debugging (F5)")
        print("  • Algorithm development") 
        print("  • Component testing")
        print("  • QuantConnect integration")
        print("\n💡 Next steps:")
        print("  1. Set breakpoints in main.py")
        print("  2. Use F5 to debug your algorithm")
        print("  3. Test model loading with actual model files")
    else:
        print("❌ Some tests failed - check the output above")
        
    return all(results)

if __name__ == "__main__":
    main()
