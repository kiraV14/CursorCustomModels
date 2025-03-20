from flask import Flask, request, jsonify, make_response
import requests
import os
import json
import logging
from waitress import serve
import subprocess
import time
import sys
from flask_cors import CORS
import time
import uuid
import random
import traceback
from cachetools import TTLCache  # Add this import

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_RAW_DATA = os.environ.get("LOG_RAW_DATA", "1") == "1"  # Set to "0" to disable raw data logging
MAX_CHUNKS_TO_LOG = int(os.environ.get("MAX_CHUNKS_TO_LOG", "20"))  # Maximum number of chunks to log
LOG_TRUNCATE_LENGTH = int(os.environ.get("LOG_TRUNCATE_LENGTH", "1000"))  # Length to truncate logs

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("proxy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a special logger for raw request/response data that only goes to console
raw_logger = logging.getLogger("raw_data")
raw_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - RAW_DATA - %(message)s'))
raw_logger.addHandler(console_handler)
raw_logger.propagate = False  # Don't propagate to root logger

# Function to log raw data with clear formatting
def log_raw_data(title, data, truncate=LOG_TRUNCATE_LENGTH):
    """Log raw data with clear formatting and optional truncation"""
    # Skip logging if raw data logging is disabled
    if not LOG_RAW_DATA:
        return
        
    try:
        if isinstance(data, dict) or isinstance(data, list):
            formatted_data = json.dumps(data, indent=2)
        else:
            formatted_data = str(data)
        
        if truncate and len(formatted_data) > truncate:
            formatted_data = formatted_data[:truncate] + f"... [truncated, total length: {len(formatted_data)}]"
        
        separator = "=" * 40
        raw_logger.info(f"\n{separator}\n{title}\n{separator}\n{formatted_data}\n{separator}")
    except Exception as e:
        raw_logger.error(f"Error logging raw data: {str(e)}")

# Add a function to collect streaming chunks
def collect_streaming_chunks(chunks, max_chunks=MAX_CHUNKS_TO_LOG):
    """
    Collect streaming chunks into a single string for logging
    
    Parameters:
    chunks (list): List of streaming chunks
    max_chunks (int): Maximum number of chunks to include
    
    Returns:
    str: A formatted string with all chunks
    """
    if not chunks:
        return "No chunks collected"
    
    # Limit the number of chunks to avoid excessive logging
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        truncated_message = f"\n... [truncated, {len(chunks) - max_chunks} more chunks]"
    else:
        truncated_message = ""
    
    # Format the chunks
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"Chunk {i+1}:\n{chunk}")
    
    return "\n\n".join(formatted_chunks) + truncated_message

app = Flask(__name__)
# Enable CORS for all routes and origins with more permissive settings
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
         "expose_headers": ["X-Request-ID", "openai-organization", "openai-processing-ms", "openai-version"],
         "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"]
     }}
)

# Groq API key - replace with your actual key or set as environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

# OpenAI API endpoints that we'll intercept
OPENAI_CHAT_ENDPOINT = "/v1/chat/completions"
CURSOR_CHAT_ENDPOINT = "/chat/completions"  # Additional endpoint for Cursor

# Groq API endpoints
GROQ_BASE_URL = "https://api.groq.com/openai"
GROQ_CHAT_ENDPOINT = "/v1/chat/completions"

# Model mapping - map OpenAI models to Groq models
MODEL_MAPPING = {
    "gpt-4o": "qwen-2.5-coder-32b",
    "gpt-4o-2024-08-06": "qwen-2.5-coder-32b",  # Handle specific model version
    "default": "qwen-2.5-coder-32b",
    "gpt-3.5-turbo": "qwen-2.5-coder-32b"  # Add more model mappings
}

# Create a TTL cache for request deduplication (5 second TTL)
request_cache = TTLCache(maxsize=1000, ttl=5)

# Initialize a cache for storing R1 reasoning results
# TTL of 1800 seconds (30 minutes) should be sufficient for a conversation
r1_reasoning_cache = TTLCache(maxsize=100, ttl=1800)

# Add a streaming tracker to prevent multiple streaming for the same request
streaming_tracker = TTLCache(maxsize=100, ttl=10)  # 10 second TTL

# Add at the top with other constants
GROQ_TIMEOUT = 120  # 120 seconds timeout for Groq API calls
MAX_RETRIES = 3    # Maximum number of retries for failed requests

# Constants for agent mode
AGENT_MODE_ENABLED = True
AGENT_INSTRUCTIONS = """
<tool_calling>
You have tools at your disposal to solve the coding task. Follow these rules regarding tool calls:
1. ALWAYS follow the tool call schema exactly as specified and provide all necessary parameters.
2. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided.
3. **NEVER refer to tool names when speaking to the USER.** For example, instead of saying 'I need to use the edit_file tool to edit your file', just say 'I will edit your file'.
4. Only calls tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.
5. Before calling each tool, first explain to the USER why you are calling it.
6. NEVER recursively apply the same code block multiple times. If a code edit fails to apply correctly, try ONCE with a different approach or ask the user for guidance.
7. After attempting to edit a file, DO NOT repeat the same edit again if it doesn't work. Instead, explain the issue to the user and ask for guidance.
8. NEVER repeatedly attempt the same code edit multiple times. If an edit fails to apply after one retry, STOP and ask the user for guidance.
</tool_calling>

<making_code_changes>
When making code changes, NEVER output code to the USER, unless requested. Instead use one of the code edit tools to implement the change.
Use the code edit tools at most once per turn.
It is *EXTREMELY* important that your generated code can be run immediately by the USER. To ensure this, follow these instructions carefully:
1. Always group together edits to the same file in a single edit file tool call, instead of multiple smaller calls.
2. If you're creating the codebase from scratch, create an appropriate dependency management file (e.g. requirements.txt) with package versions and a helpful README.
3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.
4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER and are very expensive.
5. Unless you are appending some small easy to apply edit to a file, or creating a new file, you MUST read the the contents or section of what you're editing before editing it.
6. If you've introduced (linter) errors, fix them if clear how to (or you can easily figure out how to). Do not make uneducated guesses. And DO NOT loop more than 3 times on fixing linter errors on the same file. On the third time, you should stop and ask the user what to do next.
7. If you've suggested a reasonable code_edit that wasn't followed by the apply model, you should try reapplying the edit ONLY ONCE. If it still fails, explain the issue to the user and ask for guidance.
8. NEVER repeatedly attempt the same edit multiple times. If an edit fails to apply after one retry, STOP and ask the user for guidance.
</making_code_changes>

<searching_and_reading>
You have tools to search the codebase and read files. Follow these rules regarding tool calls:
1. If available, heavily prefer the semantic search tool to grep search, file search, and list dir tools.
2. If you need to read a file, prefer to read larger sections of the file at once over multiple smaller calls.
3. If you have found a reasonable place to edit or answer, do not continue calling tools. Edit or answer from the information you have found.
</searching_and_reading>

<preventing_recursion>
You must NEVER get stuck in a loop of repeatedly trying to apply the same code edit. If you notice that you're attempting to make the same edit more than once:
1. STOP immediately
2. Explain to the user that you were unable to apply the edit
3. Describe what you were trying to do
4. Ask the user for guidance on how to proceed
5. Wait for the user's response before taking any further action
</preventing_recursion>
"""

# Initialize a cache to track recent code edits (key: hash of edit, value: count)
# TTL of 300 seconds (5 minutes) should be enough to prevent recursive edits in a single conversation
code_edit_cache = TTLCache(maxsize=100, ttl=300)

# Track consecutive edits to the same file
file_edit_counter = TTLCache(maxsize=50, ttl=600)  # 10 minutes TTL
MAX_CONSECUTIVE_EDITS = 3  # Maximum number of consecutive edits to the same file

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    # Add CORS headers to every response
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    response.headers.add('Access-Control-Max-Age', '86400')  # 24 hours
    
    # Only log response status and minimal headers for reduced verbosity
    if response.status_code != 200:
        logger.info(f"Response status: {response.status}")
    
    # Log raw response data only for non-streaming responses
    try:
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' not in content_type:
            response_data = response.get_data(as_text=True)
            # Only log if it's not too large
            if len(response_data) < 5000:
                log_raw_data(f"RESPONSE (Status: {response.status_code})", response_data)
            else:
                # Just log a summary for large responses
                log_raw_data(f"RESPONSE (Status: {response.status_code})", 
                            f"Large response ({len(response_data)} bytes) with content type: {content_type}")
    except Exception as e:
        raw_logger.error(f"Error logging response: {str(e)}")
    
    return response

@app.route('/debug', methods=['GET'])
def debug():
    """Return debug information"""
    return jsonify({
        "status": "running",
        "endpoints": [
            "/v1/chat/completions",
            "/chat/completions",
            "/<path>/chat/completions",
            "/direct",
            "/simple",
            "/agent"
        ],
        "models": list(MODEL_MAPPING.keys()),
        "groq_api_key_set": bool(GROQ_API_KEY),
        "agent_mode_enabled": AGENT_MODE_ENABLED
    })

def format_openai_response(groq_response, original_model):
    """Format Groq response with minimal transformation"""
    try:
        # Start with the original response
        response = groq_response.copy()
        
        # Ensure only required fields exist and have correct format
        if "choices" in response and len(response["choices"]) > 0:
            for choice in response["choices"]:
                if "message" in choice and "role" not in choice["message"]:
                    choice["message"]["role"] = "assistant"
        
        # Ensure basic required fields exist
        if "created" not in response:
            response["created"] = int(time.time())
        if "object" not in response:
            response["object"] = "chat.completion"
        
        logger.info(f"Passing through Groq response with minimal transformation")
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a basic response structure
        return {
            "object": "chat.completion",
            "created": int(time.time()),
            "model": original_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi!"
                    },
                    "finish_reason": "stop"
                }
            ]
        }

# Handle OPTIONS requests for all routes
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle OPTIONS requests for all routes"""
    logger.info(f"OPTIONS request: /{path}")
    
    # Create a response with all the necessary CORS headers
    response = make_response('')
    response.status_code = 200
    
    # Add all the headers that Cursor might expect
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin')
    response.headers.add('Access-Control-Expose-Headers', 'X-Request-ID, openai-organization, openai-processing-ms, openai-version')
    response.headers.add('Access-Control-Max-Age', '86400')  # 24 hours
    
    return response

def process_chat_request():
    """Process a chat completion request from any endpoint"""
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info but less verbosely
        logger.info(f"Request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Check if we're already streaming this request
        request_hash = hash(str(request.data))
        if request_hash in streaming_tracker:
            logger.warning(f"Detected duplicate streaming request (hash: {request_hash}). Returning empty response.")
            # Return a simple response to prevent recursive streaming
            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen-2.5-coder-32b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Request already being processed. Please wait for the response."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
        
        # Mark this request as being streamed
        streaming_tracker[request_hash] = True
        
        # Log raw request data
        log_raw_data("REQUEST HEADERS", dict(request.headers))
        
        if request.is_json:
            log_raw_data("REQUEST JSON BODY", request.json)
        else:
            log_raw_data("REQUEST RAW BODY", request.data.decode('utf-8', errors='replace'))
        
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            logger.info("OPTIONS preflight request")
            return handle_options(request.path.lstrip('/'))
        
        # Get the request data
        if request.is_json:
            data = request.json
            # Log message count and types without full content
            if 'messages' in data:
                messages = data['messages']
                msg_summary = [f"{m.get('role', 'unknown')}: {len(m.get('content', ''))}" for m in messages]
                logger.info(f"Processing {len(messages)} messages: {msg_summary}")
                
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Get R1 reasoning for all requests
            r1_reasoning = None
            try:
                cache_key = json.dumps(data, sort_keys=True)
                if cache_key in r1_reasoning_cache:
                    logger.info("Using cached R1 reasoning")
                    r1_reasoning = r1_reasoning_cache[cache_key]
                    logger.info(f"Retrieved cached reasoning of length: {len(r1_reasoning)}")
                else:
                    logger.info("No cached reasoning found, proceeding with R1 call")
                    
                    # Create R1 request with focus on reasoning chain
                    r1_request = {
                        "model": "deepseek-r1-distill-qwen-32b",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a reasoning chain generator. Your task is to analyze the user's request and create a structured reasoning chain that follows this format:

<reasoning_chain>
1. CONTEXT ANALYSIS
- Available files and their purposes
- Current state and issues
- User's specific request

2. IMPLEMENTATION APPROACH
- Required changes
- Potential challenges
- Dependencies and considerations

3. EXECUTION PLAN
- Step-by-step implementation
- Testing requirements
- Success criteria

4. VALIDATION STRATEGY
- Error handling
- Edge cases
- Quality assurance steps
</reasoning_chain>

Focus ONLY on creating this reasoning chain. DO NOT provide any implementation details or code."""
                            }
                        ],
                        "temperature": 0.3,  # Lower temperature for more deterministic reasoning
                        "max_tokens": 1000,
                        "stream": False  # Never stream the R1 request
                    }
                    
                    # Add user messages but filter out assistant messages
                    user_messages = [msg for msg in messages if msg['role'] in ['user', 'system']]
                    r1_request['messages'].extend(user_messages)
                    
                    # Send request to R1
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {GROQ_API_KEY}"
                    }
                    
                    log_raw_data("R1 REQUEST", r1_request)
                    
                    r1_response_raw = requests.post(
                        f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                        json=r1_request,
                        headers=headers,
                        timeout=GROQ_TIMEOUT
                    )
                    
                    logger.info(f"R1 response status: {r1_response_raw.status_code}")
                    log_raw_data("R1 RAW RESPONSE", r1_response_raw.text)
                    
                    if r1_response_raw.status_code == 200:
                        r1_response = r1_response_raw.json()
                        log_raw_data("R1 PARSED RESPONSE", r1_response)
                        
                        if 'choices' in r1_response and len(r1_response['choices']) > 0:
                            r1_reasoning = r1_response['choices'][0]['message']['content']
                            logger.info(f"Successfully extracted reasoning chain (length: {len(r1_reasoning)})")
                            r1_reasoning_cache[cache_key] = r1_reasoning
            except Exception as e:
                logger.error(f"Error getting reasoning from R1: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue without reasoning
                r1_reasoning = None
            
            # Add R1 reasoning to system message if available
            if r1_reasoning:
                has_system = False
                for msg in messages:
                    if msg.get('role') == 'system':
                        has_system = True
                        # Append reasoning to existing system message
                        msg['content'] += f"\n\nReasoning chain:\n{r1_reasoning}"
                        break
                
                if not has_system:
                    # Insert a system message with the reasoning at the beginning
                    messages.insert(0, {
                        "role": "system",
                        "content": f"Reasoning chain:\n{r1_reasoning}"
                    })
            
            # Map to Groq model if needed
            if 'model' in data:
                model = data['model']
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
                logger.info(f"Non-JSON request parsed for model: {data.get('model', 'unknown')}")
            except:
                logger.error("Failed to parse request data")
                data = {}
        
        # Check cache for this exact request
        cache_key = None
        if request.is_json:
            try:
                cache_key = json.dumps(data, sort_keys=True)
                if cache_key in request_cache:
                    logger.info("Using cached response for duplicate request")
                    return request_cache[cache_key]
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
        
        # Always enable streaming for better reliability
        request_data = data.copy()
        request_data['stream'] = True
        request_data['model'] = groq_model
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending streaming request to Groq with {len(request_data.get('messages', []))} messages")
        log_raw_data("GROQ REQUEST", request_data)
        
        def generate():
            try:
                # Create a list to collect streaming chunks for logging
                collected_chunks = []
                
                # Track if we're in a code block to prevent premature closing
                in_code_block = False
                code_block_count = 0
                last_chunk_time = time.time()
                
                with requests.post(
                    f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                    json=request_data,
                    headers=headers,
                    stream=True,
                    timeout=GROQ_TIMEOUT
                ) as groq_response:
                    
                    # Check for error status
                    if groq_response.status_code != 200:
                        error_msg = groq_response.text[:200] if hasattr(groq_response, 'text') else "Unknown error"
                        logger.error(f"Groq API error: {groq_response.status_code} - {error_msg}")
                        error_response = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": data.get('model', 'unknown'),
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"**Error: {error_msg}**\n\nPlease try a different approach or ask the user for guidance."
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        }
                        
                        log_raw_data("GROQ ERROR RESPONSE", error_response)
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Process the streaming response
                    for line in groq_response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            last_chunk_time = time.time()
                            
                            # Collect the chunk for logging
                            collected_chunks.append(line)
                            
                            # Check if we're entering or exiting a code block
                            if line.startswith('data: ') and '"content":"```' in line:
                                in_code_block = True
                                code_block_count += 1
                                logger.info(f"Entering code block #{code_block_count}")
                            elif line.startswith('data: ') and '"content":"```' in line and in_code_block:
                                in_code_block = False
                                logger.info(f"Exiting code block #{code_block_count}")
                            
                            if line.startswith('data: '):
                                # Pass through the streaming data without model name modification
                                yield f"{line}\n\n"
                            elif line.strip() == 'data: [DONE]':
                                yield "data: [DONE]\n\n"
                                return  # Ensure we exit the generator after [DONE]
                    
                    # Log all collected chunks at once
                    if collected_chunks:
                        log_raw_data("GROQ STREAMING RESPONSE (COMPLETE)", 
                                    collect_streaming_chunks(collected_chunks))
                    
                    # If we were in a code block, make sure we send a proper closing
                    if in_code_block:
                        logger.info("Detected unclosed code block, sending closing marker")
                        # Send a dummy chunk to keep the connection alive
                        dummy_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": groq_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": ""},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(dummy_chunk)}\n\n"
                    
                    # Always send a final [DONE] marker
                    yield "data: [DONE]\n\n"
                    
                    # Wait a moment before closing to ensure all data is processed
                    time.sleep(0.5)

            except requests.exceptions.Timeout:
                logger.error("Groq API timeout")
                error_response = {
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout_error",
                        "code": "timeout"
                    }
                }
                log_raw_data("TIMEOUT ERROR", error_response)
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "stream_error"
                    }
                }
                log_raw_data("STREAMING ERROR", {"error": str(e), "traceback": traceback.format_exc()})
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Clear the cache after processing the request
                if cache_key in request_cache:
                    del request_cache[cache_key]
                    logger.info("Cache cleared for request")
                
                # Remove this request from the streaming tracker
                if request_hash in streaming_tracker:
                    del streaming_tracker[request_hash]
                    logger.info(f"Removed request from streaming tracker (hash: {request_hash})")

        # Return a streaming response with proper headers (removing Connection: keep-alive)
        response = app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                # Remove 'Connection': 'keep-alive' header - it's a hop-by-hop header not allowed in WSGI
                'access-control-expose-headers': 'X-Request-ID',
                'x-request-id': request_id
            }
        )
        
        logger.info(f"Started streaming response (request ID: {request_id})")
        return response
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        error_response = make_response(jsonify(error_response_data))
        error_response.status_code = 500
        error_response.headers.add('Content-Type', 'application/json')
        
        return error_response

# Route for standard OpenAI endpoint
@app.route(OPENAI_CHAT_ENDPOINT, methods=['POST', 'OPTIONS'])
def openai_chat_completions():
    logger.info(f"Request to standard OpenAI endpoint")
    return process_chat_request()

# Route for Cursor's custom endpoint
@app.route(CURSOR_CHAT_ENDPOINT, methods=['POST', 'OPTIONS'])
def cursor_chat_completions():
    logger.info(f"Request to Cursor endpoint")
    return process_chat_request()

# Catch-all route for any other chat completions endpoint
@app.route('/<path:path>/chat/completions', methods=['POST', 'OPTIONS'])
def any_chat_completions(path):
    logger.info(f"Request to custom path: /{path}/chat/completions")
    return process_chat_request()

# Add a route for OpenAI's models endpoint
@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def list_models():
    """Return a fake list of models"""
    logger.info("Request to models endpoint")
    
    models = [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "gpt-4o-2024-08-06",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "default",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai"
        }
    ]
    
    # Create response with OpenAI-specific headers
    response = make_response(jsonify({"data": models, "object": "list"}))
    
    # Add OpenAI specific headers (avoiding hop-by-hop headers)
    response.headers.add('access-control-expose-headers', 'X-Request-ID')
    response.headers.add('openai-organization', 'user-68tm5q5hm9sro0tao3xi5e9i')
    response.headers.add('openai-processing-ms', '10')
    response.headers.add('openai-version', '2020-10-01')
    response.headers.add('strict-transport-security', 'max-age=15724800; includeSubDomains')
    response.headers.add('x-request-id', str(uuid.uuid4()))
    
    # Set correct Content-Type header
    response.headers.set('Content-Type', 'application/json')
    
    return response

# Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Return health status of the proxy server"""
    logger.info("Health check request")
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time
    })

@app.route('/', methods=['GET'])
def home():
    logger.info("Home page request")
    return """
    <html>
    <head>
        <title>Groq Proxy Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            .endpoint { background-color: #e0f7fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Groq Proxy Server is running</h1>
        <p>This server proxies requests to the following endpoints:</p>
        <div class="endpoint">
            <h3>/v1/chat/completions (Standard OpenAI endpoint)</h3>
            <p>Use this endpoint for standard OpenAI API compatibility</p>
        </div>
        <div class="endpoint">
            <h3>/chat/completions (Cursor endpoint)</h3>
            <p>Use this endpoint for Cursor compatibility</p>
        </div>
        <div class="endpoint">
            <h3>/direct (Direct endpoint)</h3>
            <p>Simple endpoint that takes a single message and returns a response</p>
        </div>
        <div class="endpoint">
            <h3>/simple (Simple endpoint)</h3>
            <p>Simple non-streaming endpoint with OpenAI-compatible response format</p>
        </div>
        <div class="endpoint">
            <h3>/agent (Agent mode endpoint)</h3>
            <p>Endpoint with agent mode instructions included in system prompt</p>
        </div>
        
        <h2>Test the API</h2>
        <p>You can test the API with the following curl command:</p>
        <pre>
curl -X POST \\
  [YOUR_NGROK_URL]/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a test assistant."},
      {"role": "user", "content": "Testing. Just say hi and nothing else."}
    ]
  }'
        </pre>
        
        <h2>Test Agent Mode</h2>
        <p>You can test the agent mode with the following curl command:</p>
        <pre>
curl -X POST \\
  [YOUR_NGROK_URL]/agent \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Please help me understand how to use the agent mode."}
    ]
  }'
        </pre>
        
        <h2>Debug Information</h2>
        <p>For debug information, visit <a href="/debug">/debug</a></p>
        <p>For health check, visit <a href="/health">/health</a></p>
    </body>
    </html>
    """

def start_ngrok(port):
    """Start ngrok and return the public URL"""
    try:
        # Check if ngrok is installed
        try:
            subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ngrok is not installed or not in PATH. Please install ngrok first.")
            print("ngrok is not installed or not in PATH. Please install ngrok first.")
            print("Visit https://ngrok.com/download to download and install ngrok")
            sys.exit(1)
            
        # Start ngrok with recommended settings for Cursor
        logger.info(f"Starting ngrok on port {port}...")
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Started ngrok process (PID: {ngrok_process.pid})")
        
        # Wait for ngrok to start
        logger.info("Waiting for ngrok to initialize...")
        time.sleep(3)
        
        # Get the public URL from ngrok API
        try:
            logger.info("Requesting tunnel information from ngrok API...")
            response = requests.get("http://localhost:4040/api/tunnels")
            tunnels = response.json()["tunnels"]
            if tunnels:
                # Using https tunnel is recommended for Cursor
                https_tunnels = [t for t in tunnels if t["public_url"].startswith("https")]
                if https_tunnels:
                    public_url = https_tunnels[0]["public_url"]
                else:
                    public_url = tunnels[0]["public_url"]
                
                logger.info(f"ngrok public URL: {public_url}")
                
                print(f"\n{'='*60}")
                print(f"NGROK PUBLIC URL: {public_url}")
                print(f"NGROK INSPECTOR: http://localhost:4040")
                print(f"Use this URL in Cursor as your OpenAI API base URL")
                print(f"{'='*60}\n")
                
                # Print example PowerShell command
                print("Example PowerShell command to test the proxy:")
                print(f"""
$headers = @{{
    "Content-Type" = "application/json"
    "Authorization" = "Bearer fake-api-key"
}}

$body = @{{
    "messages" = @(
        @{{
            "role" = "system"
            "content" = "You are a test assistant."
        }},
        @{{
            "role" = "user"
            "content" = "Testing. Just say hi and nothing else."
        }}
    )
    "model" = "gpt-4o"
}} | ConvertTo-Json

Invoke-WebRequest -Uri "{public_url}/v1/chat/completions" -Method Post -Headers $headers -Body $body
                """)
                
                # Print curl command for testing
                print("\nOr use this curl command:")
                print(f"""
curl -X POST \\
  {public_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer fake-api-key" \\
  -d '{{
    "model": "gpt-4o",
    "messages": [
      {{"role": "system", "content": "You are a test assistant."}},
      {{"role": "user", "content": "Testing. Just say hi and nothing else."}}
    ]
  }}'
                """)
                
                # Print instructions for Cursor
                print("\nTo configure Cursor:")
                print(f"1. Set the OpenAI API base URL to: {public_url}")
                print("2. Use any OpenAI model name that Cursor supports")
                print("3. Set any API key (it won't be checked)")
                print("4. Check the ngrok inspector at http://localhost:4040 to debug traffic")
                
                return public_url
            else:
                logger.error("No ngrok tunnels found")
                print("No ngrok tunnels found. Please check ngrok configuration.")
                return None
        except Exception as e:
            logger.error(f"Error getting ngrok URL: {str(e)}")
            print(f"Error getting ngrok URL: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error starting ngrok: {str(e)}")
        print(f"Error starting ngrok: {str(e)}")
        return None

# Store app start time for uptime tracking
start_time = time.time()

# Add a new direct endpoint for simple message passing
@app.route('/direct', methods=['POST', 'OPTIONS'])
def direct_completion():
    """Simple endpoint that takes a single message and returns a response"""
    logger.info("Request to direct endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('direct')
    
    try:
        # Get the request data
        if request.is_json:
            data = request.json
            message = data.get('message', '')
            model = data.get('model', 'qwen-2.5-coder-32b')
            logger.info(f"Direct request for model: {model}")
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
                message = data.get('message', '')
                model = data.get('model', 'qwen-2.5-coder-32b')
            except:
                logger.error("Failed to parse direct request data")
                return jsonify({"error": "Invalid request format"}), 400
        
        # Create a simple request to Groq
        groq_request = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": False  # No streaming for direct endpoint
        }
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending direct request to Groq")
        log_raw_data("DIRECT REQUEST", groq_request)
        
        response = requests.post(
            f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
            json=groq_request,
            headers=headers,
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
            log_raw_data("DIRECT ERROR RESPONSE", response.text)
            return jsonify({
                "error": f"Groq API error: {response.status_code}",
                "message": "Failed to get response from Groq"
            }), response.status_code
        
        # Parse the response
        log_raw_data("DIRECT RAW RESPONSE", response.text)
        groq_response = response.json()
        log_raw_data("DIRECT PARSED RESPONSE", groq_response)
        
        # Extract just the content from the response
        if "choices" in groq_response and len(groq_response["choices"]) > 0:
            content = groq_response["choices"][0]["message"]["content"]
            result = {"response": content}
            log_raw_data("DIRECT FINAL RESPONSE", result)
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error processing direct request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Add a simple non-streaming endpoint for Cursor
@app.route('/simple', methods=['POST', 'OPTIONS'])
def simple_completion():
    """Simple non-streaming endpoint for Cursor"""
    logger.info("Request to simple endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('simple')
    
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info
        logger.info(f"Simple request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Get the request data
        if request.is_json:
            data = request.json
            # Log message count without full content
            if 'messages' in data:
                messages = data['messages']
                msg_count = len(messages)
                logger.info(f"Processing {msg_count} messages in simple mode")
                
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Log model information
            if 'model' in data:
                model = data['model']
                logger.info(f"Simple request for model: {model}")
                # Map to Groq model if needed
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            logger.error("Failed to parse request data")
            return jsonify({"error": "Invalid request format"}), 400
        
        # Create request for Groq
        groq_request = data.copy()
        groq_request['model'] = groq_model
        groq_request['stream'] = False  # Explicitly disable streaming
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending non-streaming request to Groq")
        log_raw_data("SIMPLE REQUEST", groq_request)
        
        response = requests.post(
            f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
            json=groq_request,
            headers=headers,
            timeout=GROQ_TIMEOUT
        )
        
        if response.status_code != 200:
            logger.error(f"Groq API error: {response.status_code} - {response.text[:200]}")
            log_raw_data("SIMPLE ERROR RESPONSE", response.text)
            return jsonify({
                "error": {
                    "message": f"Groq API error: {response.status_code}",
                    "type": "server_error",
                    "code": "groq_error"
                }
            }), response.status_code
        
        # Parse the response
        log_raw_data("SIMPLE RAW RESPONSE", response.text)
        groq_response = response.json()
        log_raw_data("SIMPLE PARSED RESPONSE", groq_response)
        
        # Format as OpenAI response
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,  # Use the original model name
            "choices": groq_response.get("choices", []),
            "usage": groq_response.get("usage", {})
        }
        
        log_raw_data("SIMPLE FORMATTED RESPONSE", openai_response)
        logger.info(f"Successfully processed simple request")
        return jsonify(openai_response)
            
    except Exception as e:
        logger.error(f"Error processing simple request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        return jsonify(error_response_data), 500

def handle_model_instructions(model_name, instructions):
    """
    This function demonstrates how to handle model instructions and tool usage.
    
    Parameters:
    model_name (str): The name of the model (e.g., 'qwen-2.5-coder-32b')
    instructions (str): The instructions for the model
    
    Returns:
    dict: A dictionary containing the response
    """
    logger.info(f"Handling instructions for model: {model_name}")
    
    # Process the instructions
    response = {
        "model": model_name,
        "instructions": instructions,
        "processed": True,
        "timestamp": time.time()
    }
    
    # Log the response
    logger.info(f"Processed instructions for model: {model_name}")
    
    return response

def handle_tool_usage(tool_name, parameters):
    """
    This function demonstrates how to handle tool usage for models.
    
    Parameters:
    tool_name (str): The name of the tool to use
    parameters (dict): The parameters for the tool
    
    Returns:
    dict: A dictionary containing the response
    """
    logger.info(f"Using tool: {tool_name} with parameters: {parameters}")
    
    # Check for recursive code edits if this is an edit_file tool
    if tool_name == "edit_file" and "code_edit" in parameters:
        # Create a simple hash of the edit to use as a cache key
        edit_hash = hash(parameters.get("code_edit", ""))
        target_file = parameters.get("target_file", "")
        cache_key = f"{target_file}:{edit_hash}"
        
        # Check if we've seen this edit before
        if cache_key in code_edit_cache:
            # Increment the count
            code_edit_cache[cache_key] += 1
            count = code_edit_cache[cache_key]
            
            # If we've seen this edit more than twice, return an error
            if count > 2:
                logger.warning(f"Detected recursive code edit attempt ({count} times) for {target_file}")
                return {
                    "error": True,
                    "message": "Recursive code edit detected. Please try a different approach or ask the user for guidance.",
                    "tool": tool_name,
                    "parameters": parameters,
                    "timestamp": time.time()
                }
        else:
            # First time seeing this edit
            code_edit_cache[cache_key] = 1
        
        # Track consecutive edits to the same file
        if target_file in file_edit_counter:
            file_edit_counter[target_file] += 1
        else:
            file_edit_counter[target_file] = 1
        
        # Check if we've exceeded the maximum number of consecutive edits
        if file_edit_counter[target_file] > MAX_CONSECUTIVE_EDITS:
            logger.warning(f"Exceeded maximum consecutive edits ({MAX_CONSECUTIVE_EDITS}) for {target_file}")
            return {
                "error": True,
                "message": f"You've made {file_edit_counter[target_file]} consecutive edits to {target_file}. Please take a step back and reconsider your approach or ask the user for guidance.",
                "tool": tool_name,
                "parameters": parameters,
                "timestamp": time.time()
            }
    elif tool_name != "edit_file":
        # Reset the consecutive edit counter for all files if we're using a different tool
        file_edit_counter.clear()
    
    # Process the tool usage
    response = {
        "tool": tool_name,
        "parameters": parameters,
        "processed": True,
        "timestamp": time.time()
    }
    
    # Log the response
    logger.info(f"Processed tool usage: {tool_name}")
    
    return response

def send_request_to_groq(request_data):
    """
    Send a request to Groq API and return the response
    
    Parameters:
    request_data (dict): The request data to send to Groq
    
    Returns:
    dict: The response from Groq
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    # Log the request
    logger.info(f"Sending request to Groq for model: {request_data.get('model', 'unknown')}")
    
    # Try up to MAX_RETRIES times
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                json=request_data,
                headers=headers,
                timeout=GROQ_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Groq API error (attempt {attempt+1}/{MAX_RETRIES}): {response.status_code} - {response.text[:200]}")
                if attempt == MAX_RETRIES - 1:
                    raise Exception(f"Groq API error: {response.status_code} - {response.text[:200]}")
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error sending request to Groq (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # This should never be reached due to the exception in the loop
    raise Exception("Failed to send request to Groq after multiple attempts")

def extract_content_from_response(response):
    """
    Extract the content from a Groq API response
    
    Parameters:
    response (dict): The response from Groq
    
    Returns:
    str: The content from the response
    """
    if not response.get('choices'):
        raise Exception("No choices in response")
    
    return response['choices'][0]['message']['content']

@app.route('/agent', methods=['POST', 'OPTIONS'])
def agent_mode():
    """Special agent mode endpoint that includes agent instructions in the system prompt"""
    logger.info("Request to agent mode endpoint")
    
    if request.method == 'OPTIONS':
        return handle_options('agent')
    
    try:
        # Get client IP (for logging purposes)
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Log the request info
        logger.info(f"Agent request from {client_ip} using {user_agent.split(' ')[0]}")
        
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Check if we're already streaming this request
        request_hash = hash(str(request.data))
        if request_hash in streaming_tracker:
            logger.warning(f"Detected duplicate streaming request (hash: {request_hash}). Returning empty response.")
            # Return a simple response to prevent recursive streaming
            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen-2.5-coder-32b",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Request already being processed. Please wait for the response."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
        
        # Mark this request as being streamed
        streaming_tracker[request_hash] = True
        
        # Log raw request data
        log_raw_data("REQUEST HEADERS", dict(request.headers))
        
        if request.is_json:
            log_raw_data("REQUEST JSON BODY", request.json)
        else:
            log_raw_data("REQUEST RAW BODY", request.data.decode('utf-8', errors='replace'))
        
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            logger.info("OPTIONS preflight request")
            return handle_options(request.path.lstrip('/'))
        
        # Get the request data
        if request.is_json:
            data = request.json
            # Log message count and types without full content
            if 'messages' in data:
                messages = data['messages']
                msg_summary = [f"{m.get('role', 'unknown')}: {len(m.get('content', ''))}" for m in messages]
                logger.info(f"Processing {len(messages)} messages: {msg_summary}")
                
                # Take only the last few messages if there are too many
                if len(messages) > 10:
                    logger.info(f"Truncating message history from {len(messages)} to last 10 messages")
                    # Always include the system message if present
                    system_messages = [m for m in messages if m.get('role') == 'system']
                    other_messages = [m for m in messages if m.get('role') != 'system']
                    
                    # Keep system messages and last 9 other messages
                    truncated_messages = system_messages + other_messages[-9:]
                    data['messages'] = truncated_messages
                    logger.info(f"Truncated to {len(truncated_messages)} messages")
            
            # Get R1 reasoning for all requests
            r1_reasoning = None
            try:
                cache_key = json.dumps(data, sort_keys=True)
                if cache_key in r1_reasoning_cache:
                    logger.info("Using cached R1 reasoning")
                    r1_reasoning = r1_reasoning_cache[cache_key]
                    logger.info(f"Retrieved cached reasoning of length: {len(r1_reasoning)}")
                else:
                    logger.info("No cached reasoning found, proceeding with R1 call")
                    
                    # Create R1 request with focus on reasoning chain
                    r1_request = {
                        "model": "deepseek-r1-distill-qwen-32b",
                        "messages": [
                            {
                                "role": "system",
                                "content": """You are a reasoning chain generator. Your task is to analyze the user's request and create a structured reasoning chain that follows this format:

<reasoning_chain>
1. CONTEXT ANALYSIS
- Available files and their purposes
- Current state and issues
- User's specific request

2. IMPLEMENTATION APPROACH
- Required changes
- Potential challenges
- Dependencies and considerations

3. EXECUTION PLAN
- Step-by-step implementation
- Testing requirements
- Success criteria

4. VALIDATION STRATEGY
- Error handling
- Edge cases
- Quality assurance steps
</reasoning_chain>

Focus ONLY on creating this reasoning chain. DO NOT provide any implementation details or code."""
                            }
                        ],
                        "temperature": 0.3,  # Lower temperature for more deterministic reasoning
                        "max_tokens": 1000,
                        "stream": False  # Never stream the R1 request
                    }
                    
                    # Add user messages but filter out assistant messages
                    user_messages = [msg for msg in messages if msg['role'] in ['user', 'system']]
                    r1_request['messages'].extend(user_messages)
                    
                    # Send request to R1
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {GROQ_API_KEY}"
                    }
                    
                    log_raw_data("R1 REQUEST", r1_request)
                    
                    r1_response_raw = requests.post(
                        f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                        json=r1_request,
                        headers=headers,
                        timeout=GROQ_TIMEOUT
                    )
                    
                    logger.info(f"R1 response status: {r1_response_raw.status_code}")
                    log_raw_data("R1 RAW RESPONSE", r1_response_raw.text)
                    
                    if r1_response_raw.status_code == 200:
                        r1_response = r1_response_raw.json()
                        log_raw_data("R1 PARSED RESPONSE", r1_response)
                        
                        if 'choices' in r1_response and len(r1_response['choices']) > 0:
                            r1_reasoning = r1_response['choices'][0]['message']['content']
                            logger.info(f"Successfully extracted reasoning chain (length: {len(r1_reasoning)})")
                            r1_reasoning_cache[cache_key] = r1_reasoning
            except Exception as e:
                logger.error(f"Error getting reasoning from R1: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue without reasoning
                r1_reasoning = None
            
            # Add R1 reasoning to system message if available
            if r1_reasoning:
                has_system = False
                for msg in messages:
                    if msg.get('role') == 'system':
                        has_system = True
                        # Append reasoning to existing system message
                        msg['content'] += f"\n\nReasoning chain:\n{r1_reasoning}"
                        break
                
                if not has_system:
                    # Insert a system message with the reasoning at the beginning
                    messages.insert(0, {
                        "role": "system",
                        "content": f"Reasoning chain:\n{r1_reasoning}"
                    })
            
            # Add agent instructions to system message
            has_system = False
            for msg in messages:
                if msg.get('role') == 'system':
                    has_system = True
                    # Append agent instructions to existing system message if not already there
                    if AGENT_INSTRUCTIONS not in msg['content']:
                        msg['content'] += f"\n\n{AGENT_INSTRUCTIONS}"
                    break
            
            if not has_system:
                # Insert a system message with the agent instructions at the beginning
                messages.insert(0, {
                    "role": "system",
                    "content": AGENT_INSTRUCTIONS
                })
            
            # Map to Groq model if needed
            if 'model' in data:
                model = data['model']
                if model in MODEL_MAPPING:
                    groq_model = MODEL_MAPPING[model]
                else:
                    groq_model = MODEL_MAPPING["default"]
            else:
                groq_model = MODEL_MAPPING["default"]
                logger.info(f"No model specified, using default: {groq_model}")
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
                logger.info(f"Non-JSON request parsed for model: {data.get('model', 'unknown')}")
            except:
                logger.error("Failed to parse request data")
                data = {}
        
        # Check cache for this exact request
        cache_key = None
        if request.is_json:
            try:
                cache_key = json.dumps(data, sort_keys=True)
                if cache_key in request_cache:
                    logger.info("Using cached response for duplicate request")
                    return request_cache[cache_key]
            except Exception as e:
                logger.error(f"Error checking cache: {str(e)}")
        
        # Always enable streaming for better reliability
        request_data = data.copy()
        request_data['stream'] = True
        request_data['model'] = groq_model
        
        # Forward the request to Groq
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        logger.info(f"Sending agent mode request to Groq")
        log_raw_data("AGENT MODE REQUEST", request_data)
        
        def generate():
            try:
                # Create a list to collect streaming chunks for logging
                collected_chunks = []
                
                # Track if we're in a code block to prevent premature closing
                in_code_block = False
                code_block_count = 0
                last_chunk_time = time.time()
                
                with requests.post(
                    f"{GROQ_BASE_URL}{GROQ_CHAT_ENDPOINT}",
                    json=request_data,
                    headers=headers,
                    stream=True,
                    timeout=GROQ_TIMEOUT
                ) as groq_response:
                    
                    # Check for error status
                    if groq_response.status_code != 200:
                        error_msg = groq_response.text[:200] if hasattr(groq_response, 'text') else "Unknown error"
                        logger.error(f"Groq API error: {groq_response.status_code} - {error_msg}")
                        error_response = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": data.get('model', 'unknown'),
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"**Error: {error_msg}**\n\nPlease try a different approach or ask the user for guidance."
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        }
                        
                        log_raw_data("AGENT MODE ERROR RESPONSE", error_response)
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Process the streaming response
                    for line in groq_response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            last_chunk_time = time.time()
                            
                            # Collect the chunk for logging
                            collected_chunks.append(line)
                            
                            # Check if we're entering or exiting a code block
                            if line.startswith('data: ') and '"content":"```' in line:
                                in_code_block = True
                                code_block_count += 1
                                logger.info(f"Entering code block #{code_block_count}")
                            elif line.startswith('data: ') and '"content":"```' in line and in_code_block:
                                in_code_block = False
                                logger.info(f"Exiting code block #{code_block_count}")
                            
                            if line.startswith('data: '):
                                # Pass through the streaming data without model name modification
                                yield f"{line}\n\n"
                            elif line.strip() == 'data: [DONE]':
                                yield "data: [DONE]\n\n"
                                return  # Ensure we exit the generator after [DONE]
                    
                    # Log all collected chunks at once
                    if collected_chunks:
                        log_raw_data("AGENT MODE STREAMING RESPONSE (COMPLETE)", 
                                    collect_streaming_chunks(collected_chunks))
                    
                    # If we were in a code block, make sure we send a proper closing
                    if in_code_block:
                        logger.info("Detected unclosed code block, sending closing marker")
                        # Send a dummy chunk to keep the connection alive
                        dummy_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": groq_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": ""},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(dummy_chunk)}\n\n"
                    
                    # Always send a final [DONE] marker
                    yield "data: [DONE]\n\n"
                    
                    # Wait a moment before closing to ensure all data is processed
                    time.sleep(0.5)

            except requests.exceptions.Timeout:
                logger.error("Groq API timeout")
                error_response = {
                    "error": {
                        "message": "Request timeout",
                        "type": "timeout_error",
                        "code": "timeout"
                    }
                }
                log_raw_data("TIMEOUT ERROR", error_response)
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "stream_error"
                    }
                }
                log_raw_data("STREAMING ERROR", {"error": str(e), "traceback": traceback.format_exc()})
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Clear the cache after processing the request
                if cache_key in request_cache:
                    del request_cache[cache_key]
                    logger.info("Cache cleared for request")
                
                # Remove this request from the streaming tracker
                if request_hash in streaming_tracker:
                    del streaming_tracker[request_hash]
                    logger.info(f"Removed request from streaming tracker (hash: {request_hash})")

        # Return a streaming response with proper headers (removing Connection: keep-alive)
        response = app.response_class(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                # Remove 'Connection': 'keep-alive' header - it's a hop-by-hop header not allowed in WSGI
                'access-control-expose-headers': 'X-Request-ID',
                'x-request-id': request_id
            }
        )
        
        logger.info(f"Started streaming response (request ID: {request_id})")
        return response
            
    except Exception as e:
        logger.error(f"Error processing agent mode request: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a properly structured error response
        error_response_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": "no_completion"
            }
        }
        
        error_response = make_response(jsonify(error_response_data))
        error_response.status_code = 500
        error_response.headers.add('Content-Type', 'application/json')
        
        return error_response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Groq proxy server on port {port}")
    
    # Start ngrok in a separate thread
    public_url = start_ngrok(port)
    
    # Start the Flask server
    print(f"Starting Groq proxy server on port {port}")
    logger.info(f"Server starting on port {port}")
    
    try:
        serve(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.critical(f"Server failed to start: {str(e)}")
        print(f"Server failed to start: {str(e)}")
        sys.exit(1) 