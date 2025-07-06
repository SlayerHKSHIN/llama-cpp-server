#!/usr/bin/env python3
import requests
import json
import sys

# Configuration
API_KEY = "your-api-key-here"  # Replace with your actual API key
BASE_URL = "http://localhost:30000"

def test_streaming():
    """Test the streaming endpoint"""
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "user", "content": "Tell me a short story about a robot."}
        ],
        "max_tokens": 150,
        "stream": True
    }
    
    print("Testing streaming endpoint...")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        print("\n\nStream completed.")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"\nFailed to parse: {data_str}")
                        
    except Exception as e:
        print(f"Error: {e}")

def test_non_streaming():
    """Test the non-streaming endpoint"""
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50,
        "stream": False
    }
    
    print("\n\nTesting non-streaming endpoint...")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"Response: {content}")
        else:
            print("Unexpected response format:", result)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    
    test_non_streaming()
    test_streaming()