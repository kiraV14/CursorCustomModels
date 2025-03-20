import subprocess
import requests
import sys
import time
import os

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

def main():
    # Check if ngrok is running
    ngrok_url = get_ngrok_url()
    
    if not ngrok_url:
        print("Starting ngrok and proxy server...")
        # Start the proxy server in the background
        proxy_process = subprocess.Popen(
            ["python", "groq_proxy.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ngrok to start
        print("Waiting for ngrok to start...")
        for _ in range(10):  # Try for 10 seconds
            time.sleep(1)
            ngrok_url = get_ngrok_url()
            if ngrok_url:
                break
    
    if not ngrok_url:
        print("Failed to get ngrok URL. Please start the proxy server manually.")
        sys.exit(1)
    
    print(f"Found ngrok URL: {ngrok_url}")
    
    # Run the test with the correct URL
    print("\nRunning test with the correct ngrok URL...")
    test_command = f"python test_proxy.py --url {ngrok_url} --endpoint /chat/completions --model default"
    print(f"Command: {test_command}")
    
    # Run the test
    os.system(test_command)

if __name__ == "__main__":
    main() 