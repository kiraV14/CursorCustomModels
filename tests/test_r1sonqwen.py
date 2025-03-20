import requests
import json
import sys
import argparse
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_r1sonqwen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_r1sonqwen(url="http://localhost:5000", endpoint="/v1/chat/completions"):
    """Test the r1sonqwen model with a simple request"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-dummy-key"  # The proxy doesn't check this
    }
    
    # Create a simple test request
    data = {
        "model": "r1sonqwen",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant."
            },
            {
                "role": "user",
                "content": "Write a Python function to calculate the Fibonacci sequence up to n terms."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    logger.info(f"Sending request to {url}{endpoint} for model r1sonqwen...")
    logger.info("This may take a while as it needs to call two models in sequence.")
    logger.info(f"Request data: {json.dumps(data)}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{url}{endpoint}",
            headers=headers,
            json=data,
            timeout=120  # Longer timeout for the chain
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Request completed in {elapsed_time:.2f} seconds")
        
        # Log the raw response
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        logger.info(f"Raw response text: {response.text[:1000]}...")
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Print the response
                logger.info("\n=== Response ===")
                logger.info(f"Status: {response.status_code}")
                logger.info(f"Model: {result.get('model', 'unknown')}")
                
                # Print the content
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    logger.info("\n=== Content ===")
                    logger.info(f"Content length: {len(content)}")
                    logger.info(f"Content preview: {content[:500]}...")
                    
                    # Print usage information
                    if 'usage' in result:
                        usage = result['usage']
                        logger.info("\n=== Usage ===")
                        logger.info(f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                        logger.info(f"Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                        logger.info(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
                else:
                    logger.error("No content in response")
                    logger.error(f"Full response: {json.dumps(result)}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(f"Response text: {response.text}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error("Exception details:", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the r1sonqwen model")
    parser.add_argument("--url", default="http://localhost:5000", help="URL of the proxy server")
    parser.add_argument("--endpoint", default="/v1/chat/completions", help="Endpoint to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    test_r1sonqwen(args.url, args.endpoint) 