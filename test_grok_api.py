#!/usr/bin/env python3
"""Test Grok API Key"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('XAI_API_KEY')
if not api_key:
    print("No XAI_API_KEY found in .env")
    exit(1)

print(f"Testing Grok API key: {api_key[:10]}...")

# Test 1: Check models endpoint
headers = {'Authorization': f'Bearer {api_key}'}
response = requests.get('https://api.x.ai/v1/models', headers=headers)
print(f"\nModels endpoint status: {response.status_code}")
if response.status_code == 200:
    models = response.json()
    print("Available models:")
    for model in models.get('data', [])[:5]:
        print(f"  - {model['id']}")
else:
    print(f"Error: {response.text}")

# Test 2: Try a simple chat completion
url = 'https://api.x.ai/v1/chat/completions'
data = {
    'model': 'grok-3-mini-beta',
    'messages': [{'role': 'user', 'content': 'Say hello'}],
    'max_tokens': 10
}

response = requests.post(url, json=data, headers=headers)
print(f"\nChat completion status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Response: {result['choices'][0]['message']['content']}")
else:
    print(f"Error: {response.text}")
    if response.status_code == 403:
        print("\n⚠️  403 Forbidden - Possible issues:")
        print("  1. API key not activated - check https://x.ai/api")
        print("  2. Billing not set up - may need payment method")
        print("  3. Rate limited - check your usage")
        print("  4. Wrong API key - regenerate at https://x.ai/api")