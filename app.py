import os
import re
from typing import TypedDict

import faiss
import gradio as gr
import numpy as np
from google import genai
from langgraph.graph import StateGraph, START, END

# Read the API key from Render Environment Variables.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it in Render → your service → Environment."
    )

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-3.8-flash"
EMBED_MODEL = "gemini-embedding-001"

cybersecurity_documents = [
    """Phishing attacks involve fraudulent messages designed to steal
    personal information, passwords, banking credentials and OTPs.
    Users should never share OTPs through suspicious links.""",
    """Fake job scams promise unrealistic salaries and request
    registration fees, training fees or security deposits.
    Genuine employers should not demand suspicious advance payments.""",
    """Banking scams often involve urgent messages claiming that
    accounts will be blocked unless users click suspicious links.
    Banks do not ask customers to disclose their OTP or PIN.""",
    """Lottery scams inform users that they have won a prize
    but demand processing fees or taxes before releasing money.
    Unexpected prize messages should be independently verified.""",
    """Investment scams promise guaranteed high returns with little
    or no risk. Users should verify financial companies and
    avoid transferring money to unknown individuals.""",
    """Social engineering manipulates people into revealing confidential
    information through fear, urgency, impersonation or fake authority.""",
    """Suspicious URLs may contain misspelled brand names, unusual
    domains, misleading subdomains or unexpected redirects.
    HTTPS alone does not guarantee that a website is trustworthy."""
]

def generate_embedding(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return np.asarray(result.embeddings[0].values, dtype="float32")

# Build the FAISS knowledge index when the web service starts.
document_embeddings = np.asarray(
    [generate_embedding(doc) for doc in cybersecurity_documents],
    dtype="float32"
)
vector_store = faiss.IndexFlatL2(document_embeddings.shape[1])
vector_store.add(document_embeddings)

def retrieve_knowledge(query, top_k=3):
    query_vector = np.asarray([generate_embedding(query)], dtype="float32")
    _, indices = vector_store.search(
        query_vector, min(top_k, len(cybersecurity_documents))
    )
    return "\n".join(cybersecurity_documents[i] for i in indices[0])

def detect_suspicious_patterns(text):
    patterns = {
        "Urgent action": [
            "immediately", "urgent", "today", "account blocked", "act now"
        ],
        "Financial demand": [
            "registration fee", "processing fee", "security deposit", "send money"
        ],
        "Credential theft": [
            "share otp", "enter otp", "password", "bank pin", "verify your account"
        ],
        "Unrealistic promises": [
            "guaranteed profit", "won a lottery", "earn 50000", "free money"
        ]
    }
    found = []
    lowered = text.lower()
    for category, words in patterns.items():
        for word in words:
            if word in lowered:
                found.append(f"{category}: {word}")
    return list(dict.fromkeys(found))

def calculate_risk_score(patterns, url_warnings):
    # This is a transparent heuristic, not a calibrated fraud probability.
    score = min(len(patterns) * 20 + len([
        w for w in url_warnings if w != "No URL provided"
    ]) * 10, 100)
    if score >= 70:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level

def analyze_url(url):
    warnings = []
    url = (url or "").lower().strip()
    if not url:
        return ["No URL provided"]
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if re.search(r"https?://\d+\.\d+\.\d+\.\d+", url):
        warnings.append("IP address used as website")
    if "@" in url:
        warnings.append("Contains @ symbol")
    if url.count("-") >= 3:
        warnings.append("Multiple hyphens in URL")
    if url.startswith("http://"):
        warnings.append("Does not use HTTPS")
    if not warnings:
        warnings.append("No simple URL warning detected; this does not prove the site is safe")
    return warnings

class ScamState(TypedDict):
    message: str
    url: str
    patterns: list
    url_warnings: list
    knowledge: str
    score: int
    level: str
    report: str

def analyze_node(state):
    patterns = detect_suspicious_patterns(state["message"])
    url_warnings = analyze_url(state["url"])
    score, level = calculate_risk_score(patterns, url_warnings)
    return {
        "patterns": patterns,
        "url_warnings": url_warnings,
        "score": score,
        "level": level
    }

def retrieval_node(state):
    knowledge = retrieve_knowledge(state["message"] + " " + state["url"])
    return {"knowledge": knowledge}

def report_node(state):
    prompt = f"""
You are ScamSense AI, a cybersecurity analysis assistant.

MESSAGE: {state['message']}
URL: {state['url'] or 'Not provided'}
DETECTED PATTERNS: {state['patterns']}
URL WARNINGS: {state['url_warnings']}
HEURISTIC RISK SCORE: {state['score']}/100
RISK LEVEL: {state['level']}
RETRIEVED CYBERSECURITY KNOWLEDGE: {state['knowledge']}

Write a clear report with:
1. Risk classification and score
2. Suspicious indicators
3. Explanation of why the indicators matter
4. Practical safety recommendations

Important:
- The score is a simple heuristic, not a probability of fraud.
- Do not claim a website is verified, safe, or malicious.
- Distinguish suspicious indicators from confirmed fraud.
- If no indicators are found, say that no obvious indicators were detected,
  not that the message or website is guaranteed safe.
"""
    response = client.interactions.create(model=MODEL, input=prompt)
    return {"report": response.output_text}

workflow = StateGraph(ScamState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("report", report_node)
workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "report")
workflow.add_edge("report", END)
scam_agent = workflow.compile()

def scam_sense(message, url):
    message = (message or "").strip()
    url = (url or "").strip()
    if not message and not url:
        return "Please enter a message or URL."
    if not message:
        message = "Please analyze this URL: " + url
    try:
        result = scam_agent.invoke({
            "message": message,
            "url": url,
            "patterns": [],
            "url_warnings": [],
            "knowledge": "",
            "score": 0,
            "level": "",
            "report": ""
        })
        return result["report"]
    except Exception as exc:
        return f"Analysis failed: {type(exc).__name__}: {exc}"

with gr.Blocks(theme=gr.themes.Soft(), title="ScamSense AI") as demo:
    gr.Markdown(
        """
        # 🛡️ ScamSense AI
        ### Intelligent Scam Detection Agent

        Analyze suspicious SMS messages, emails, job offers and URLs.
        This tool provides risk indicators and safety guidance; it does not
        perform live threat-intelligence checks or guarantee fraud detection.
        """
    )
    message_input = gr.Textbox(
        label="Suspicious message (optional if URL is entered)",
        placeholder="Paste an SMS, email or job offer...",
        lines=5
    )
    url_input = gr.Textbox(
        label="Website URL (optional)",
        placeholder="https://example.com"
    )
    analyze_button = gr.Button("🔍 Analyze for Scams", variant="primary")
    output = gr.Markdown(label="Analysis Report")
    analyze_button.click(
        fn=scam_sense,
        inputs=[message_input, url_input],
        outputs=output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
