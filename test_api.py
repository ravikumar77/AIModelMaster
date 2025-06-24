#!/usr/bin/env python3
"""
Test script for LLM Platform API endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000/api"

def test_api():
    print("Testing LLM Platform API...")
    
    # Test health check
    print("\n1. Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test list models
    print("\n2. Testing list models...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    models = response.json()['models']
    print(f"Found {len(models)} models")
    
    if models:
        model_id = models[0]['id']
        print(f"Using model ID: {model_id}")
        
        # Test get specific model
        print(f"\n3. Testing get model {model_id}...")
        response = requests.get(f"{BASE_URL}/models/{model_id}")
        print(f"Status: {response.status_code}")
        print(f"Model: {response.json()['name']}")
        
        # Test text generation
        print(f"\n4. Testing text generation...")
        gen_data = {
            "prompt": "The future of artificial intelligence is",
            "temperature": 0.7,
            "max_length": 50
        }
        response = requests.post(f"{BASE_URL}/models/{model_id}/generate", json=gen_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['text']}")
            print(f"Generation time: {result['generation_time']:.2f}s")
        
        # Test create training job
        print(f"\n5. Testing create training job...")
        training_data = {
            "model_id": model_id,
            "job_name": "API Test Training",
            "epochs": 2,
            "learning_rate": 0.001,
            "batch_size": 4
        }
        response = requests.post(f"{BASE_URL}/training", json=training_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 201:
            job = response.json()
            job_id = job['id']
            print(f"Training job created: {job['job_name']} (ID: {job_id})")
            
            # Test get training job
            print(f"\n6. Testing get training job...")
            time.sleep(2)  # Wait a bit for training to start
            response = requests.get(f"{BASE_URL}/training/{job_id}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                job_info = response.json()
                print(f"Job status: {job_info['status']}")
                print(f"Progress: {job_info['progress']:.1f}%")
        
        # Test model evaluation
        print(f"\n7. Testing model evaluation...")
        eval_data = {
            "eval_name": "API Test Evaluation"
        }
        response = requests.post(f"{BASE_URL}/models/{model_id}/evaluate", json=eval_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 201:
            eval_result = response.json()
            print(f"Evaluation completed: {eval_result['eval_name']}")
            print(f"Metrics: {eval_result['metrics']}")
    
    # Test list training jobs
    print(f"\n8. Testing list training jobs...")
    response = requests.get(f"{BASE_URL}/training")
    print(f"Status: {response.status_code}")
    jobs = response.json()['training_jobs']
    print(f"Found {len(jobs)} training jobs")
    
    # Test statistics
    print(f"\n9. Testing statistics...")
    response = requests.get(f"{BASE_URL}/statistics")
    print(f"Status: {response.status_code}")
    stats = response.json()
    print(f"Platform stats: {stats}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    test_api()