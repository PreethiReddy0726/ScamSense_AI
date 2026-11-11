# ScamSense AI

A Gradio-based scam-awareness assistant using Gemini, LangGraph and a FAISS
retrieval knowledge base.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in your environment.
3. Run: `python app.py`

## Render deployment

Create a Render Web Service from this repository. The `render.yaml` file
provides the build and start commands. Add `GEMINI_API_KEY` as an Environment
Variable in the Render service settings.

The risk score is a simple heuristic, not a calibrated fraud probability.
The app does not perform live URL reputation checks and cannot guarantee
whether a message or website is fraudulent.
