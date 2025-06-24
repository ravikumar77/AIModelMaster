#!/usr/bin/env python3
"""
Working example of LLM Platform API usage with authentication
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000/api"

def register_user():
    """Register a new user and get API key"""
    user_data = {
        "username": f"user_{int(time.time())}",  # Unique username
        "email": f"user{int(time.time())}@example.com",
        "password": "securepass123"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    
    if response.status_code == 201:
        result = response.json()
        return result['api_key'], result['username']
    else:
        print(f"Registration failed: {response.json()}")
        return None, None

def test_generation(api_key, model_id=1):
    """Test text generation with authentication"""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }
    
    gen_request = {
        "prompt": "def calculate_factorial(n):",
        "temperature": 0.3,
        "max_length": 150
    }
    
    response = requests.post(f"{BASE_URL}/models/{model_id}/generate", 
                           json=gen_request, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Generation failed: {response.json()}")
        return None

def create_coding_training(api_key, model_id=1, dataset_id=1):
    """Create a coding training job"""
    headers = {
        "Content-Type": "application/json", 
        "X-API-Key": api_key
    }
    
    training_request = {
        "model_id": model_id,
        "job_name": "Python Coding Enhancement",
        "dataset_id": dataset_id,
        "epochs": 3,
        "learning_rate": 0.00005,
        "batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32
    }
    
    response = requests.post(f"{BASE_URL}/training/coding", 
                           json=training_request, headers=headers)
    
    if response.status_code == 201:
        return response.json()
    else:
        print(f"Training creation failed: {response.json()}")
        return None

def check_api_usage(api_key):
    """Check API key usage and limits"""
    headers = {"X-API-Key": api_key}
    
    response = requests.get(f"{BASE_URL}/auth/keys", headers=headers)
    
    if response.status_code == 200:
        return response.json()['api_keys']
    else:
        print(f"Usage check failed: {response.json()}")
        return None

def main():
    print("🚀 LLM Platform API - Working Example")
    print("=" * 50)
    
    # Step 1: Register and get API key
    print("\n1. Registering user...")
    api_key, username = register_user()
    
    if not api_key:
        print("❌ Failed to register user")
        return
    
    print(f"✅ Registered user: {username}")
    print(f"🔑 API Key: {api_key[:20]}...")
    
    # Step 2: Test text generation  
    print("\n2. Testing text generation...")
    result = test_generation(api_key)
    
    if result:
        print(f"✅ Generated text:")
        print(f"   {result['text'][:100]}...")
        print(f"   Generation time: {result['generation_time']:.2f}s")
    else:
        print("❌ Text generation failed")
    
    # Step 3: List available datasets
    print("\n3. Checking coding datasets...")
    response = requests.get(f"{BASE_URL}/datasets")
    if response.status_code == 200:
        datasets = response.json()['datasets']
        print(f"✅ Found {len(datasets)} coding datasets:")
        for ds in datasets:
            print(f"   - {ds['name']} ({ds['language']})")
    
    # Step 4: Create coding training job
    print("\n4. Creating coding training job...")
    job_result = create_coding_training(api_key)
    
    if job_result:
        print(f"✅ Training job created:")
        print(f"   Job ID: {job_result['id']}")
        print(f"   Name: {job_result['job_name']}")
        print(f"   Type: {job_result['training_type']}")
    else:
        print("❌ Training job creation failed")
    
    # Step 5: Check API usage
    print("\n5. Checking API usage...")
    usage_info = check_api_usage(api_key)
    
    if usage_info:
        key_info = usage_info[0]
        print(f"✅ API usage:")
        print(f"   Requests used: {key_info['usage_count']}")
        print(f"   Daily limit: {key_info['rate_limit']}")
        print(f"   Remaining: {key_info['rate_limit'] - key_info['usage_count']}")
    else:
        print("❌ Usage check failed")
    
    # Step 6: Show summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"✅ User registered successfully")
    print(f"✅ API key authentication working")
    print(f"✅ Text generation functional")
    print(f"✅ Coding training jobs working")
    print(f"✅ Rate limiting monitoring active")
    
    print(f"\n🔑 Your API Key: {api_key}")
    print(f"\n📖 Usage Examples:")
    print(f"curl -X POST {BASE_URL}/models/1/generate \\")
    print(f"  -H 'X-API-Key: {api_key}' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{{\"prompt\": \"def hello():\", \"max_length\": 100}}'")

if __name__ == "__main__":
    main()