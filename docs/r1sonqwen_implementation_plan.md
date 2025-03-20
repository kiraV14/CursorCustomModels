# R1sonQwen Implementation Plan

## Overview

This document outlines the implementation plan for the "r1sonqwen" model, which creates a workflow/prompt chain between the Deepseek R1 model and Qwen. The workflow follows these steps:

1. Receive the original request from Cursor
2. Send the request to Deepseek R1 model (deepseek-r1-distill-qwen-32b) for reasoning
3. Take the reasoning output and combine it with the original request
4. Send the combined prompt to Qwen for final code generation
5. Return the Qwen response to Cursor

## Implementation Details

### 1. Model Configuration

Add the new model to the `MODEL_MAPPING` dictionary:

```python
MODEL_MAPPING = {
    "gpt-4o": "qwen-2.5-coder-32b",
    "gpt-4o-2024-08-06": "qwen-2.5-coder-32b",
    "default": "qwen-2.5-coder-32b",
    "gpt-3.5-turbo": "qwen-2.5-coder-32b",
    "r1sonqwen": "custom_chain"  # Special identifier for our custom chain
}
```

### 2. Create a Memory Cache for Storing Intermediate Results

We'll need a cache to store the intermediate reasoning results from the R1 model:

```python
# Initialize a cache for storing R1 reasoning results
# TTL of 1800 seconds (30 minutes) should be sufficient for a conversation
r1_reasoning_cache = TTLCache(maxsize=100, ttl=1800)
```

### 3. Create a New Endpoint for the R1sonQwen Chain

We'll create a new function to handle the R1sonQwen chain:

```python
def process_r1sonqwen_request(data):
    """
    Process a request using the R1sonQwen chain:
    1. Send to R1 for reasoning
    2. Combine reasoning with original request
    3. Send to Qwen for final response
    """
    # Step 1: Extract the original request
    original_messages = data.get('messages', [])
    
    # Step 2: Create a modified request for R1 reasoning
    r1_request = data.copy()
    r1_request['model'] = "deepseek-r1-distill-qwen-32b"
    
    # Add a system message to instruct R1 to focus on reasoning
    r1_system_message = {
        "role": "system",
        "content": "You are an expert reasoning assistant. Your task is to analyze the user's request and provide detailed reasoning about how to approach the problem. Focus on breaking down the problem, identifying key components, and outlining a solution strategy. Do NOT provide any code implementation, just the reasoning process."
    }
    
    # Check if there's already a system message
    has_system = any(msg.get('role') == 'system' for msg in original_messages)
    if has_system:
        # Modify the existing system message
        for i, msg in enumerate(r1_request['messages']):
            if msg.get('role') == 'system':
                r1_request['messages'][i]['content'] += "\n\n" + r1_system_message['content']
    else:
        # Add a new system message
        r1_request['messages'].insert(0, r1_system_message)
    
    # Step 3: Send the request to R1 for reasoning
    r1_response = send_request_to_groq(r1_request)
    
    # Step 4: Extract the reasoning from R1's response
    r1_reasoning = extract_content_from_response(r1_response)
    
    # Step 5: Create a new request for Qwen with the reasoning included
    qwen_request = data.copy()
    qwen_request['model'] = "qwen-2.5-coder-32b"
    
    # Add the reasoning as a system message for Qwen
    qwen_system_message = {
        "role": "system",
        "content": f"You are an expert coding assistant. Use the following reasoning to guide your response, but focus on implementing the solution:\n\n### REASONING FROM R1:\n{r1_reasoning}\n\nBased on this reasoning, provide a complete and working implementation that addresses the user's request."
    }
    
    # Check if there's already a system message
    has_system = any(msg.get('role') == 'system' for msg in original_messages)
    if has_system:
        # Modify the existing system message
        for i, msg in enumerate(qwen_request['messages']):
            if msg.get('role') == 'system':
                qwen_request['messages'][i]['content'] += "\n\n" + qwen_system_message['content']
    else:
        # Add a new system message
        qwen_request['messages'].insert(0, qwen_system_message)
    
    # Step 6: Send the request to Qwen for final response
    qwen_response = send_request_to_groq(qwen_request)
    
    # Step 7: Return the Qwen response
    return qwen_response
```

### 4. Helper Functions

We'll need some helper functions:

```python
def send_request_to_groq(request_data):
    """Send a request to Groq API and return the response"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    response = requests.post(
        f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
        json=request_data,
        headers=headers,
        timeout=GROQ_TIMEOUT
    )
    
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code} - {response.text[:200]}")
    
    return response.json()

def extract_content_from_response(response):
    """Extract the content from a Groq API response"""
    if not response.get('choices'):
        raise Exception("No choices in response")
    
    return response['choices'][0]['message']['content']
```

### 5. Modify the Request Processing Logic

We'll need to modify the `process_chat_request` function to handle the r1sonqwen model:

```python
def process_chat_request():
    # ... existing code ...
    
    # Check if the model is r1sonqwen
    if 'model' in data and data['model'] == 'r1sonqwen':
        logger.info("Processing r1sonqwen request")
        return process_r1sonqwen_request(data)
    
    # ... rest of the existing code ...
```

## Caching Strategy

To improve performance and reduce API calls:

1. Cache the R1 reasoning results using the request as the key
2. Use a TTL of 30 minutes for the cache
3. If the same request comes in again, reuse the cached reasoning

## Error Handling

1. If the R1 model fails, fall back to using just Qwen
2. If Qwen fails, return an appropriate error message
3. Log all errors for debugging

## Testing Plan

1. Test with simple coding requests
2. Test with complex multi-step problems
3. Compare results with and without the R1 reasoning step
4. Measure response times and optimize as needed

## Future Improvements

1. Add a parameter to control the depth of reasoning
2. Allow users to provide their own reasoning instructions
3. Implement a feedback mechanism to improve the chain
4. Add support for streaming responses from both models 