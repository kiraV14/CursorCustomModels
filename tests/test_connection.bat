@echo off
echo Testing Groq Proxy Connection...
python test_proxy.py --endpoint /chat/completions --model default
pause 