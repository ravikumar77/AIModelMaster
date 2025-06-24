#!/usr/bin/env python3
"""
Test script for authentication and coding training APIs
"""
import requests
import json

BASE_URL = "http://localhost:5000/api"

def test_auth_and_coding():
    print("Testing Authentication and Coding Features...")
    
    # Test user registration
    print("\n1. Testing user registration...")
    user_data = {
        "username": "testcoder",
        "email": "test@coder.com", 
        "password": "securepass123"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    print(f"Registration Status: {response.status_code}")
    if response.status_code == 201:
        reg_result = response.json()
        api_key = reg_result['api_key']
        print(f"API Key: {api_key}")
    else:
        # Use demo key if registration fails
        api_key = "llm_demo_key_12345678901234567890"
        print(f"Using demo API key: {api_key}")
    
    headers = {"X-API-Key": api_key}
    
    # Test list API keys
    print("\n2. Testing API key listing...")
    response = requests.get(f"{BASE_URL}/auth/keys", headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        keys = response.json()['api_keys']
        print(f"Found {len(keys)} API keys")
    
    # Test coding datasets
    print("\n3. Testing coding datasets...")
    response = requests.get(f"{BASE_URL}/datasets")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        datasets = response.json()['datasets']
        print(f"Found {len(datasets)} coding datasets")
        for dataset in datasets:
            print(f"  - {dataset['name']} ({dataset['language']})")
    
    # Test text generation with API key
    print("\n4. Testing authenticated text generation...")
    gen_data = {
        "prompt": "def fibonacci(n):",
        "temperature": 0.3,
        "max_length": 100
    }
    response = requests.post(f"{BASE_URL}/models/1/generate", json=gen_data, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Generated code: {result['text'][:100]}...")
    
    # Test coding training job
    if datasets:
        print("\n5. Testing coding training job...")
        coding_training_data = {
            "model_id": 1,
            "job_name": "Python Coding Training",
            "dataset_id": datasets[0]['id'],
            "epochs": 3,
            "learning_rate": 0.00005,
            "batch_size": 4
        }
        response = requests.post(f"{BASE_URL}/training/coding", json=coding_training_data, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 201:
            job = response.json()
            print(f"Coding training job created: {job['job_name']}")
            print(f"Training type: {job['training_type']}")
    
    print("\nAuthentication and coding API testing completed!")

if __name__ == "__main__":
    test_auth_and_coding()