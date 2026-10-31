# Cisco Nexus 9000 Troubleshooting Assistant

A small Retrieval-Augmented Generation (RAG) demo that indexes the Cisco Nexus 9000 troubleshooting PDF locally, retrieves the five most relevant passages, and uses Groq's GPT-OSS 120B model to answer questions with page-aware context.

## Requirements

- Python 3.11
- Internet access for the first local embedding-model download and Groq inference
- A configured `GROQ_API_KEY` in the project `.env` file
- The guide at `data/cisco_nexus_9000_troubleshooting_guide.pdf`

**The Cisco PDF is not included in this repository.** Obtain a copy you are permitted to use and place it at `data/cisco_nexus_9000_troubleshooting_guide.pdf` before running the app.

## Setup (Windows PowerShell)

Run these commands from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
```

The printed executable should be inside this project's `.venv` directory. Add `GROQ_API_KEY=...` to a local `.env` file; never commit or share that file. `.env.example` is a placeholder template.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open the local Gradio URL printed in the terminal. The first run downloads the CPU embedding model and creates the persistent `chroma_db/` index. Later runs reuse that index. If the PDF is replaced, stop the app, remove the project `chroma_db` directory, and start the app again to rebuild it.

## Smoke tests

Run the app's two real Groq-backed example questions:

```powershell
python app.py --smoke-test
```

Run the focused offline unit tests:

```powershell
python -m unittest -v
```

The direct lookup example asks what to do when a port remains in a "Link Failure or Not Connected" state; that condition is documented in the supplied guide. The broader example compares the guide's link-failure and unexpected-link-flapping guidance. Verify answers and page references against the guide during a demonstration.

## Troubleshooting

- **PDF not found:** place the guide at `data/cisco_nexus_9000_troubleshooting_guide.pdf`.
- **Missing API key:** confirm `GROQ_API_KEY` is set in `.env`.
- **No PDF text:** scanned/image-only PDFs need OCR, which this POC does not include.
- **Embedding download/init failure:** check internet access and free disk space, then retry.
- **Groq/API error:** check internet access, the key, account model availability, and terminal logs.
- **Stale index:** delete `chroma_db/` and restart to re-index the current guide.
