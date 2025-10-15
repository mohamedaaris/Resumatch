"""
Test script to verify basic functionality without research dependencies
"""

import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported without research dependencies"""
    try:
        from app import app, initialize_model
        print("✓ app module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import app module: {e}")
        return False
    
    try:
        from enhanced_model import EnhancedResuMatchModel
        print("✓ enhanced_model module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import enhanced_model module: {e}")
        return False
    
    try:
        from enhanced_preprocess import EnhancedTextPreprocessor
        print("✓ enhanced_preprocess module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import enhanced_preprocess module: {e}")
        return False
    
    try:
        from gnn_recommender import create_gnn_recommender
        print("✓ gnn_recommender module imported successfully")
    except Exception as e:
        print(f"✓ gnn_recommender module import failed (expected without PyTorch): {e}")
    
    try:
        from rl_recommender import create_rl_recommender
        print("✓ rl_recommender module imported successfully")
    except Exception as e:
        print(f"✓ rl_recommender module import failed (expected without PyTorch): {e}")
    
    try:
        from bandit_recommender import create_bandit_recommender
        print("✓ bandit_recommender module imported successfully")
    except Exception as e:
        print(f"✓ bandit_recommender module import failed (expected without dependencies): {e}")
    
    try:
        from ensemble_recommender import create_ensemble_recommender
        print("✓ ensemble_recommender module imported successfully")
    except Exception as e:
        print(f"✓ ensemble_recommender module import failed (expected without dependencies): {e}")
    
    return True

def test_model_initialization():
    """Test that the model can be initialized without research dependencies"""
    try:
        from app import initialize_model
        # This will show warnings but should not fail
        initialize_model()
        print("✓ Model initialization completed (warnings expected without research dependencies)")
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_basic_routes():
    """Test that basic routes work"""
    try:
        from app import app
        with app.test_client() as client:
            # Test home page
            response = client.get('/')
            assert response.status_code == 200
            print("✓ Home page route works")
            
            # Test research page (should work even without dependencies)
            response = client.get('/research')
            # Might return 500 if templates are missing, but shouldn't crash
            print(f"✓ Research page route responded with status {response.status_code}")
            
        return True
    except Exception as e:
        print(f"✗ Basic routes test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing ResuMatch basic functionality...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    print()
    success &= test_model_initialization()
    print()
    success &= test_basic_routes()
    
    print()
    if success:
        print("🎉 All tests passed! The application can run without research dependencies.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("\nNote: Warnings about missing research dependencies are expected and normal.")
    print("To use research features, install the optional dependencies:")
    print("pip install torch torch-geometric")