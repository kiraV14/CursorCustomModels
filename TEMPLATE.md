# CursorCustomModels - TEMPLATE

Due to Cursor's recent Max API changes, I'm publicly releasing my custom implementation allowing you to use ANY model with Cursor at your own cost with unlimited tool requests.

## What Is This?

A proxy server that lets you use alternative AI models with Cursor IDE:
- 🚀 Full Cursor compatibility with **ANY** AI provider
- 💰 Only pay for tokens you use (no subscription)
- 🔧 Unlimited tool calls
- 🔄 Works with Groq, Anthropic, Google, local models, etc.

My specific implementation combines **Deepseek's R1 model** for reasoning with **Qwen** for output generation via Groq. This combo delivers excellent performance at reasonable cost.

## Quick Setup

1. Clone: `git clone https://github.com/rinadelph/CursorCustomModels.git`
2. Install: `pip install -r requirements.txt`
3. Configure: Copy `.env.template` to `.env` and add your API keys
4. Run: `python src/multi_ai_proxy.py`
5. Connect Cursor (see below)

## Cursor Setup (CRITICAL!)

Cursor requires initial verification with a real OpenAI API key:

1. Enter a **real OpenAI key** in Cursor settings
2. Click "Verify" - this unlocks Custom API Mode
3. After verification, change "Base URL" to:
   - Local: `http://localhost:8000`
   - Remote: Your NGROK URL
4. Click "Verify" again to test your proxy
5. Select your model and start using!

## Technical Notes

- Creates a local proxy that intercepts Cursor's OpenAI-bound requests
- Routes requests to your preferred AI provider
- Includes NGROK for remote access if needed
- Streaming responses for real-time interaction
- Proper tool handling for file editing, search, etc.

## Disclaimer

This is a **proof of concept** with **no maintenance guarantees**. Use at your own risk and in accordance with all services' terms of use.

I've been using this setup for months with substantial cost savings compared to subscriptions. Feel free to fork, modify, and improve!

---

Star the repo if useful: [https://github.com/rinadelph/CursorCustomModels](https://github.com/rinadelph/CursorCustomModels)
