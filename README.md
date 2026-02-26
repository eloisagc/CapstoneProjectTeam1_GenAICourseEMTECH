# UABC Academic Q&A Assistant (v9)

## What It Does

Retrieval-Augmented Generation (RAG) assistant that answers questions about
AI policies and guidelines at UABC (Universidad Autonoma de Baja California).
Uses GPT-4o-mini via LangChain with ChromaDB vector store and a professional
Streamlit interface with real-time metrics, explainability, and user feedback.

## Team 1 Members

- **Garcia Canseco Eloisa del Carmen**
- **Inzunza Gonzalez Everardo**
- **Navarro Cota Christian Xavier**
- **Rivera Aguirre Flavio Abel**

## Version History

| Version | Focus |
|---------|-------|
| v1 | Basic RAG pipeline (Module 15) |
| v2 | Evaluation metrics + source display (Module 16) |
| v3 | Professional 3-column layout redesign |
| v4 | Pure HTML metrics, expanded settings, no flicker |
| v5 | Feedback relocation, static examples, HTML latency bars |
| v6 | GitHub-ready: sidebar reorder, custom avatars, nuclear CSS fix, repo files |
| v7 | Enhanced Settings: dynamic model/temperature, response length |
| v8 | Institutional branding (UABC logo) + team member credits |
| **v9** | **Feedback submit button + "Respuesta enviada" confirmation** |

## What's New in v9

| Feature | v8 | v9 |
|---------|----|-----|
| Comment Submit | Auto-submit on Enter (no button) | **Explicit "📩 Send Comment" button** |
| Confirmation | No feedback to user | **"✅ Respuesta enviada" message** |
| Validation | No check | **Warns if comment is empty** |
| All else | ✅ | **Identical — zero functional changes** |

## Setup Instructions

### Requirements
- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements_v9.txt
```

### API Key

```bash
export OPENAI_API_KEY="your-key-here"
```

## Running the App

### Google Colab (Recommended)

1. Upload `Team1_project_15_16_17_18_v9.ipynb` to Colab
2. Upload 3 PDF files to `/content/`
3. Upload `streamlit_app_v9.py`, `requirements_v9.txt`, and `uabc_logo.png` to `/content/`
4. Run all cells (ensure the Streamlit launch cell references `streamlit_app_v9.py`)
5. Last cell creates a public URL via localtunnel or ngrok

### Local

```bash
streamlit run streamlit_app_v9.py
```

## Architecture

```
PDFs (3 docs, 41 pages)
    --> PyPDFLoader
    --> RecursiveCharacterTextSplitter (900 chars, 150 overlap)
    --> OpenAI text-embedding-3-small
    --> ChromaDB
            |
User Query --> Retriever (k=2-8, configurable)
            --> Prompt Template (ES/EN)
            --> GPT-4o-mini / GPT-4o (selectable)
            --> Answer + Citations
```

## UI Layout (v9)

```
+------------------+-------------------------+--------------------+
|    SIDEBAR       |      CHAT CENTER        |   RIGHT PANEL      |
|                  |                         |                    |
| [UABC Logo]     | [Header Banner]         | Real-Time Metrics  |
| 🎓 UABC AI Asst |                         |  Queries | Latency |
| Module 17 - v9  | [Chat Input]            |  Satisf. | Success |
| --------------- |                         |  Feedback | Errors |
| Navigation      |                         |                    |
| --------------- | [Chat History]          | SLA Badge          |
| Example Qs      |  🧑‍💻 User messages     |                    |
|  (static text)  |  🤖 Assistant responses  | Explainability     |
| --------------- |  (scrollable 520px)     |  Confidence bar    |
| ⚙️ Settings     |                         |  Model | Temp      |
|  🌐 Language    |                         |                    |
|  🧠 Model       |                         | Retrieved Sources  |
|  🌡️ Temperature |                         |  Source chips      |
|  📄 Top-K       |                         |                    |
|  📏 Length      |                         | Latency per Query  |
|  📎 Sources     |                         |  (HTML mini-bars)  |
| --------------- |                         |                    |
| Status line     |                         | Rate Last Response |
| [Clear Chat]    |                         |  👍 👎 buttons     |
|                  |                         |  Comment text box  |
|                  |                         |  [📩 Send Comment] |
|                  |                         |  ✅ Respuesta      |
+------------------+-------------------------+--------------------+
```

## Test Results (Module 16 Evaluation)

| Metric | Value |
|--------|-------|
| Questions evaluated | 10 |
| Average latency | ~3.4 seconds |
| Citations present | 100% |
| Average unique sources per query | 2.5 |
| Off-topic rejection | 3/3 (100%) |

## What We Learned

1. **RAG grounds the LLM effectively** — Restricting responses to retrieved document chunks
   avoids hallucination. 100% of evaluated responses included verifiable citations.

2. **Prompt engineering matters more than model size** — Adding explicit fallback instructions
   significantly improved off-topic question handling without changing the model.

3. **Chunk size and overlap affect quality** — Increasing `chunk_overlap` from 80 to 150
   improved context coherence in retrieved passages.

4. **Simple metrics reveal a lot** — Tracking latency, citation presence, and source
   count was enough to identify the main issues and verify fixes.

5. **Error handling is essential for demos** — try/except blocks and loading indicators
   transformed the app from fragile to demo-ready.

6. **UI layout matters for usability** — Moving metrics to a dedicated right panel
   eliminated scrolling and made monitoring visible alongside conversation.

7. **Avoid Streamlit containers for metrics** — Pure HTML rendering eliminates the
   grey flickering boxes that Streamlit's container system causes on re-renders.

8. **Feedback placement matters** — Placing feedback in the right panel instead of
   below the chat input keeps the chat area clean and avoids visual overlap.

9. **Sidebar information architecture** — Putting example questions before settings
   guides new users to start interacting immediately.

10. **Explicit submit buttons improve UX** — Auto-submitting text inputs on Enter
    confuses users; an explicit button with confirmation message provides clear feedback.

## Known Limitations

- The system only knows what is in the 3 uploaded PDFs (41 pages, ~97 chunks)
- Latency of ~3-4 seconds per query depends on OpenAI API response time
- No conversation memory across questions (each query is independent)
- ChromaDB is ephemeral in Colab (rebuilds each session)
- Confidence score uses a simplified formula, not actual model confidence
- Feature importance values in Explainability page are illustrative (not computed dynamically)

## Future Improvements

- Add conversation memory using `ConversationBufferMemory`
- Implement actual SHAP/LIME explainability analysis
- Deploy permanently to Streamlit Cloud or Hugging Face Spaces
- Add more documents to the knowledge base
- Implement user authentication for production use

## Repository Structure

```
Modulo_16_17_18_v9/
├── streamlit_app_v9.py                 # Main Streamlit application (969 lines)
├── uabc_logo.png                       # UABC institutional logo
├── Team1_project_15_16_17_18_v9.ipynb  # Jupyter notebook (RAG pipeline + evaluation)
├── requirements_v9.txt                 # Python dependencies
├── README_v9.md                        # This file
├── CHANGELOG.md                        # Version history with detailed changes
├── LICENSE                             # MIT License
└── .gitignore                          # Git ignore rules
```

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Team 1** — Garcia Canseco, Inzunza Gonzalez, Navarro Cota, Rivera Aguirre | Modules 15-16-17-18 Capstone Project | UABC - Formacion IAG
