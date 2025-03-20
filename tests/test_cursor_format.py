import requests
import json
import sys
import argparse

def get_ngrok_url():
    """Get the ngrok public URL from the ngrok API"""
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        if tunnels:
            return tunnels[0]["public_url"]
        else:
            print("No ngrok tunnels found. Make sure ngrok is running.")
            return None
    except Exception as e:
        print(f"Error getting ngrok URL: {str(e)}")
        return None

def test_cursor_format(url):
    """Test the proxy server with Cursor's specific format"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer fake-api-key"
    }
    
    # This is the exact format Cursor uses
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a test assistant."
            },
            {
                "role": "user",
                "content": "Testing. Just say blue and nothing else."
            }
        ],
        "model": "default",
        "stream": False
    }
    
    print(f"Sending test request to: {url}/chat/completions")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(
            f"{url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Response received:")
            print(json.dumps(result, indent=2))
            
            # Check if this is a real or fake response
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if "fake response" in content.lower():
                print("\nNote: Received a fake response. This means the proxy is working but there was an issue with the Groq API call.")
            else:
                print("\nSuccess! Received a real response from Groq's Qwen model.")
            
            return True
        else:
            print(f"\nError: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"\nError connecting to proxy: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Groq proxy with Cursor's format")
    parser.add_argument("--url", default=None, help="The URL of the proxy server (default: auto-detect ngrok URL)")
    args = parser.parse_args()
    
    # Auto-detect ngrok URL if not provided
    if args.url is None or args.url == "YOUR_NGROK_URL":
        print("No URL provided, attempting to auto-detect ngrok URL...")
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            args.url = ngrok_url
            print(f"Using auto-detected ngrok URL: {args.url}")
        else:
            print("Failed to auto-detect ngrok URL. Please provide the URL manually.")
            sys.exit(1)
    
    print(f"Testing proxy at {args.url} with Cursor's format")
    success = test_cursor_format(args.url)
    
    if not success:
        sys.exit(1) 