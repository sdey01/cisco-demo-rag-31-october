# Product Requirements Document (PRD)
# Cisco Nexus 9000 Troubleshooting Guide — Basic RAG POC

**Document type:** Product Requirements Document  
**Project type:** Proof of Concept (POC)  
**Target implementation assistant:** GitHub Copilot  
**Primary language:** Python  
**UI framework:** Gradio  
**LLM inference provider:** Groq  
**LLM:** GPT-OSS 120B (`openai/gpt-oss-120b`, subject to the model identifier supported by the configured Groq account)  
**RAG framework:** LangChain  
**Vector database:** ChromaDB  
**Source document:** Cisco Nexus 9000 Series Troubleshooting Guide (PDF)

---

## 1. Instructions to GitHub Copilot

Build the application described in this document in the currently opened project/workspace directory.

Treat this PRD as the source of truth. Implement the smallest working end-to-end RAG application that satisfies the requirements. This is a customer-facing demonstration POC, not a production enterprise platform.

### Non-negotiable implementation rules

1. Use Python for the entire application.
2. Keep the code exceptionally simple and readable for learners and customer-demo audiences.
3. Use straightforward procedural code. **Do not create classes.** Avoid elaborate abstractions, factories, dependency-injection frameworks, or unnecessary helper functions.
4. Use only a few small functions where they genuinely improve readability (for example, a function to answer a question). Keep the main workflow visible and easy to follow.
5. Use LangChain for document loading, text splitting, embeddings/vector-store integration, and retrieval where practical.
6. Use ChromaDB as the vector store.
7. Retrieve the **top 5** relevant chunks for each user question.
8. Use the Groq inference API with GPT-OSS 120B as the language model. Read the API key from the existing environment file; never hard-code credentials.
9. Build the user interface with Gradio.
10. Create and use a project-local Python virtual environment named `.venv`. Install every Python dependency into that environment. **Never install project packages into global/system Python.**
11. Include a basic smoke test with two questions: one direct factual lookup and one broader conceptual question.
12. Do not add a relational database, authentication, user management, cloud deployment, agent framework, complex observability stack, or unrelated features.
13. Do not fabricate answers when the PDF does not contain sufficient evidence. The assistant must say when the answer cannot be found in the supplied document.
14. Before finishing, provide clear run instructions and report the result of the two smoke tests. Do not claim a test passed unless it was actually executed successfully.

If a detail is not specified, choose the simplest sensible implementation and document the choice. Do not stop to ask questions for minor implementation decisions.

---

## 2. Product overview

### 2.1 Problem statement

Users need to ask questions about the Cisco Nexus 9000 Series Troubleshooting Guide. Searching a lengthy technical PDF manually can be time-consuming. The POC should let a user enter a natural-language question, retrieve relevant passages from the PDF, and generate a grounded answer using an LLM.

### 2.2 Product objective

Create a minimal Retrieval-Augmented Generation (RAG) application that:

- Ingests the Cisco Nexus 9000 troubleshooting PDF.
- Extracts and splits the document into searchable text chunks.
- Generates embeddings for those chunks.
- Stores the embeddings and associated text in ChromaDB.
- Retrieves the five most relevant chunks for a user's question.
- Sends the retrieved context and question to GPT-OSS 120B through Groq.
- Displays a concise, document-grounded answer in Gradio.
- Displays the user's previous questions and answers in a history area on the left side of the interface.

### 2.3 Success criteria

The POC is successful when:

- The PDF can be loaded and indexed without manual intervention beyond placing it in the expected location.
- The vector index persists locally and can be reused on subsequent application runs.
- A user can ask a question in Gradio and receive an answer generated from retrieved PDF content.
- Exactly five chunks are requested from the retriever (`k=5`).
- The interface displays previous question-and-answer exchanges during the current application session.
- Missing or unsupported information is handled transparently.
- The two smoke-test scenarios can be executed and their outcomes reported.
- The application runs entirely from the project virtual environment.

---

## 3. Scope

### 3.1 In scope

- One Cisco Nexus 9000 troubleshooting PDF as the knowledge source.
- PDF text extraction.
- Recursive character text splitting.
- Local sentence-transformer embeddings.
- Local persistent ChromaDB vector store.
- Similarity-based retrieval of the top five chunks.
- Groq-hosted GPT-OSS 120B inference.
- Basic prompt-based RAG answer generation.
- Gradio chat interface.
- Left-side conversation history for the current session.
- Environment-variable-based API configuration.
- Local `.venv` setup and dependency installation.
- Basic smoke testing.

### 3.2 Out of scope

Do not implement the following in this POC:

- Multiple document collections or a document-management portal.
- Uploading arbitrary PDFs through the UI.
- OCR for scanned/image-only PDF pages.
- Complex table extraction or image understanding.
- Hybrid search, BM25, query expansion, reranking, or multi-stage retrieval.
- Agentic RAG, LangGraph, tool calling, or autonomous workflows.
- Multi-user authentication, authorization, or user accounts.
- Persistent conversation history across application restarts.
- SQL, MySQL, PostgreSQL, MongoDB, or any separate database layer.
- Cloud hosting, container orchestration, CI/CD, or production deployment.
- Advanced monitoring, tracing, analytics, or evaluation platforms.
- Streaming responses unless Gradio's simplest supported chat implementation makes it trivial.
- A complex frontend built separately from Python.

---

## 4. Target users and primary use case

### Target users

- Cisco Nexus 9000 administrators and network engineers.
- Technical support personnel.
- Troubleshooting and infrastructure teams.
- Customer stakeholders attending the POC demonstration.

### Primary user journey

1. The user starts the Gradio application.
2. The application loads or reuses the locally persisted ChromaDB index.
3. The user enters a natural-language question.
4. The application converts the question into an embedding.
5. The retriever searches ChromaDB and returns the top five relevant chunks.
6. The application constructs a prompt containing the retrieved context and user question.
7. GPT-OSS 120B, accessed through Groq, generates a grounded response.
8. Gradio displays the answer.
9. The question and answer are added to the left-side session history.
10. The user can ask another question.

---

## 5. Functional requirements

### FR-01: PDF knowledge source

- Use the Cisco Nexus 9000 Series Troubleshooting Guide supplied by the project owner.
- Expect the PDF to be placed in a project folder named `data/`.
- Use a clearly named file such as `data/cisco_nexus_9000_troubleshooting_guide.pdf`.
- If the actual supplied filename differs, configure it in one obvious location near the top of the application.
- Use LangChain's `PyPDFLoader` or its currently supported equivalent.
- Extract the PDF as page-level documents so page metadata can be retained.
- If the PDF is missing, show a clear error explaining where the user should place it.
- Do not silently continue with an empty knowledge base.
- OCR is not required. If the PDF contains no extractable text, report that limitation clearly.

### FR-02: Text chunking

Use LangChain's `RecursiveCharacterTextSplitter`.

Initial configuration:

| Parameter | Value |
|---|---:|
| `chunk_size` | 512 |
| `chunk_overlap` | 60 |
| `add_start_index` | True, if supported |

Requirements:

- Split the extracted PDF documents into manageable chunks.
- Preserve page metadata and any useful source metadata.
- Ensure that each chunk retains its source page information where available.
- Keep the splitting logic simple and visible in the code.

### FR-03: Embedding model

Use a lightweight local sentence-transformer embedding model:

- Recommended model: `sentence-transformers/all-MiniLM-L6-v2`.
- Use LangChain's Hugging Face embedding integration, such as `HuggingFaceEmbeddings`.
- Run embeddings locally on CPU by default.
- Do not require a GPU.
- Download the model on first use if it is not already cached.
- Reuse the same embedding model when creating and loading the vector store.

The embedding model is separate from the Groq LLM. It is used to represent PDF chunks and user questions as vectors for semantic retrieval.

### FR-04: ChromaDB vector store

- Use ChromaDB through the LangChain Chroma integration.
- Persist the vector store locally in a folder such as `chroma_db/`.
- Store chunk text, embeddings, and page/source metadata.
- On the first run, load the PDF, split it, generate embeddings, and create the persistent index.
- On later runs, reuse the existing index rather than rebuilding it every time.
- Keep the indexing workflow simple.
- If the source PDF changes, provide a straightforward documented way to rebuild the index (for example, delete the local `chroma_db/` directory and restart the application).
- Avoid introducing a separate database service.

**Important:** Do not accidentally add duplicate chunks every time the application starts. The simplest acceptable approach is to build the persistent index only when it does not already exist, and document how to rebuild it after replacing the PDF. If the chosen Chroma integration requires a more reliable existence check, use the simplest supported collection check.

### FR-05: Retriever

Use a basic similarity retriever.

Expected configuration:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

Requirements:

- Retrieve the top five semantically similar chunks.
- Do not add reranking, hybrid search, filters, or query rewriting.
- Keep `k=5` easy to locate and change.
- Return the retrieved document objects with their metadata.

### FR-06: Groq LLM integration

- Use Groq as the inference provider.
- Use the GPT-OSS 120B model supported by the configured Groq account.
- Preferred model identifier: `openai/gpt-oss-120b`. If Groq's current account/API documentation specifies a different identifier, use the supported identifier and document it.
- Use the LangChain Groq integration (`langchain-groq`) if compatible with the current model/API.
- Read the Groq API key from the existing `.env` file.
- Use `python-dotenv` to load environment variables.
- Never hard-code API keys or print their values.
- Do not create a new API key or overwrite the user's existing `.env` file.
- If the required key is absent, show a clear configuration error.

Expected environment variable:

```env
GROQ_API_KEY=your_existing_groq_api_key
```

Expected model configuration should be easy to locate, for example:

```python
MODEL_NAME = "openai/gpt-oss-120b"
```

Use a conservative temperature suitable for technical question answering (for example, `0.1` or `0.2`). Keep other generation parameters at sensible defaults unless required by the provider.

### FR-07: RAG prompt and answer generation

For each user question:

1. Retrieve the top five chunks from ChromaDB.
2. Combine the retrieved text into a context string.
3. Pass the context and question to the Groq-hosted GPT-OSS 120B model.
4. Display the model's answer.

The system prompt must instruct the model to:

- Act as a technical assistant for the Cisco Nexus 9000 Series Troubleshooting Guide.
- Answer using the supplied retrieved context as the primary and authoritative source.
- Avoid inventing commands, configuration steps, error codes, or troubleshooting procedures.
- Clearly distinguish information explicitly stated in the context from reasonable explanation.
- If the context does not contain enough information, say so clearly.
- For a broader conceptual question, synthesize relevant retrieved passages without pretending that the PDF states something it does not.
- Be concise, structured, and technically clear.
- Include relevant page references when page metadata is available.

Suggested prompt template:

```text
You are a technical assistant specializing in the Cisco Nexus 9000
Series Troubleshooting Guide.

Answer the user's question using only the retrieved document context
provided below as your primary source.

Rules:
1. Do not invent technical facts, commands, error messages, or procedures.
2. If the context does not contain enough information to answer,
   explicitly say that the supplied document context does not provide
   sufficient information.
3. For conceptual questions, synthesize the relevant retrieved
   passages, but do not claim unsupported details are in the guide.
4. Present troubleshooting steps in a clear numbered list when useful.
5. Include page references from the retrieved context when available.
6. Keep the answer focused and easy to understand.

Retrieved document context:
{context}

User question:
{question}

Answer:
```

The prompt may be adapted to the current LangChain prompt API, but preserve its intent.

### FR-08: Source/page references

- Preserve page metadata from the PDF loader through chunking and vector storage.
- Where possible, include the page number(s) supporting the answer.
- Display references in a simple format, such as `Source: Cisco Nexus 9000 Troubleshooting Guide, page 42`.
- Do not invent page numbers.
- If page metadata is missing, omit the page reference rather than guessing.
- This is a lightweight citation feature, not a full citation-management system.

### FR-09: Gradio user interface

Build a simple, clean interface using Gradio.

The interface must contain:

**Left panel — Conversation history**
- A visible section titled `Conversation History`.
- Display previous questions and their corresponding answers.
- Keep history in application memory for the current running session.
- A simple list of question/answer pairs is sufficient.
- The history should update after each successful question.
- A `Clear History` control is optional but desirable if easy to implement.

**Main panel — Chat**
- Application title: `Cisco Nexus 9000 Troubleshooting Assistant`.
- A short subtitle explaining that answers are generated from the supplied troubleshooting guide.
- A chat area showing user questions and assistant answers.
- A text box for entering a question.
- A `Submit` or `Ask` button.
- A clear/reset chat control if straightforward.
- A simple loading indication while the answer is being generated.

Use Gradio's supported chat components (for example, `gr.Blocks`, `gr.Chatbot`, `gr.Textbox`, `gr.Button`, and `gr.Markdown`) in the simplest practical layout.

A two-column layout is preferred:
- Left column: conversation history.
- Right column: chat interface.

Do not build a custom React, JavaScript, or separate web frontend.

### FR-10: Conversation history

- Store history in an in-memory Python list or Gradio session state.
- Each history entry should contain the user's question and assistant's answer.
- Display the most recent entries clearly in the left panel.
- History persistence across application restarts is not required.
- Do not add a database or file-based conversation store.
- Keep the implementation minimal.

### FR-11: Error handling

Provide understandable messages for common failures:

- PDF file not found.
- PDF contains no extractable text.
- Missing Groq API key.
- Groq API/network/model error.
- Embedding model download or initialization failure.
- ChromaDB initialization or retrieval failure.
- Empty user question.

Do not expose API keys, stack traces, or sensitive environment values in the Gradio UI. For development, errors may be logged to the terminal with enough information to troubleshoot.

---

## 6. Two question types and expected behavior

The POC should demonstrate two different kinds of user questions. The application does not need a routing classifier or separate retrieval pipelines. Both question types should use the same simple top-five retrieval flow.

### Type A — Direct factual lookup

A narrow question asking for a specific fact, command, error condition, symptom, or troubleshooting instruction stated in the PDF.

Example smoke-test question:

> According to the guide, what does the documented error or fault condition [select a specific condition present in the supplied PDF] indicate, and what troubleshooting action does the guide recommend?

Copilot must select a real, verifiable condition from the actual PDF when preparing the test. Do not invent an error code or expected answer.

Expected behavior:
- Retrieve the relevant page/chunk(s).
- Return a direct answer.
- Include a page reference when available.
- Avoid unrelated background information.

### Type B — Broader conceptual question

A question that requires combining information from more than one relevant passage.

Example smoke-test question:

> Based on the troubleshooting guide, what general approach should an engineer follow when diagnosing a recurring issue on a Cisco Nexus 9000 switch, and how should the engineer use the symptoms and troubleshooting information to narrow down the cause?

Expected behavior:
- Retrieve up to five relevant chunks.
- Synthesize the retrieved information into a coherent explanation.
- Organize the answer into practical steps or themes where useful.
- Include page references when available.
- Clearly state if the PDF does not provide enough information for part of the question.

**Important:** The broader question is intended to demonstrate synthesis from retrieved content, not open-ended answers based on unsupported model knowledge.

---

## 7. Technical architecture

### 7.1 High-level flow

```text
                 OFFLINE / STARTUP INGESTION

  Cisco Nexus 9000 Troubleshooting Guide (PDF)
                         |
                         v
                  PyPDFLoader
                         |
                         v
                 Extract page text
                         |
                         v
          RecursiveCharacterTextSplitter
                 chunk_size = 512
                 chunk_overlap = 60
                         |
                         v
             all-MiniLM-L6-v2 Embeddings
                    (local CPU)
                         |
                         v
                  ChromaDB
               (persistent local store)


                  ONLINE QUESTION FLOW

                   User question
                         |
                         v
              all-MiniLM-L6-v2 Embedding
                         |
                         v
                  ChromaDB Retriever
                  Similarity search
                      top k = 5
                         |
                         v
                Retrieved text chunks
                         |
                         v
             Prompt: context + question
                         |
                         v
                Groq Inference API
                  GPT-OSS 120B
                         |
                         v
                  Generated answer
                         |
                         v
                  Gradio Chat UI
                         |
              +----------+-----------+
              |                      |
              v                      v
        Main chat panel       Left-side history
```

### 7.2 Architectural principles

- One PDF knowledge source.
- One local embedding model.
- One persistent ChromaDB collection.
- One retriever configured for top-five similarity search.
- One Groq-hosted LLM.
- One Gradio application.
- No separate backend server is necessary.
- No external database service is necessary.
- No agent or orchestration framework is necessary.

---

## 8. Recommended project structure

Create a clean project structure similar to:

```text
cisco-nexus-rag/
│
├── .venv/                         # Project-local virtual environment
├── .env                           # Existing Groq API key; never commit
├── .env.example                   # Placeholder configuration only
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                         # Main RAG + Gradio application
│
├── data/
│   └── cisco_nexus_9000_troubleshooting_guide.pdf
│
└── chroma_db/                     # Persistent local vector index
```

Keep the application in one main file (`app.py`) if practical. Do not split a tiny POC into many modules merely for architectural formality.

The `.venv/` and `chroma_db/` directories should be excluded from version control. The `.env` file must also be excluded.

---

## 9. Python environment and dependency requirements

### 9.1 Virtual environment — mandatory

The project must use a local virtual environment named `.venv` in the project root.

**Never install dependencies into global Python.**

First inspect the available Python installation. Use a compatible Python version supported by all selected dependencies (Python 3.11 or 3.12 is a sensible default if available).

Windows PowerShell setup:

```powershell
# Run from the project root
py -3.12 -m venv .venv

# Activate the environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip inside the virtual environment
python -m pip install --upgrade pip
```

If Python 3.12 is unavailable, use another installed compatible Python version. Do not install a new global Python version without user approval.

If PowerShell blocks activation, use the appropriate activation method for the user's shell. Do not change system-wide execution policy.

### 9.2 Install packages only inside `.venv`

After activating `.venv`, install the project requirements:

```powershell
python -m pip install -r requirements.txt
```

Use `python -m pip`, not a bare global `pip` command.

Before installing, verify that Python resolves to the project's `.venv` executable:

```powershell
python -c "import sys; print(sys.executable)"
```

The printed path must point inside the project `.venv` directory.

Also verify:

```powershell
python -m pip --version
```

Its installation path must point inside `.venv`.

### 9.3 Suggested dependencies

Use compatible, current stable package versions. Pin versions in `requirements.txt` after confirming compatibility.

Likely direct dependencies:

```text
langchain
langchain-community
langchain-chroma
langchain-huggingface
langchain-groq
langchain-text-splitters
chromadb
sentence-transformers
pypdf
gr​​adio
gpython-dotenv
```

Add only packages actually required by the chosen imports. Avoid unnecessary dependencies.

If a package has been renamed or its integration has moved, use the currently supported package and import path. Update `requirements.txt` and `README.md` accordingly.

### 9.4 Environment file

The user already has an `.env` file containing the Groq API key.

- Preserve the existing `.env` file.
- Do not overwrite it.
- Do not print or expose the secret.
- Load it using `load_dotenv()`.
- Read `GROQ_API_KEY` from the environment.
- If the actual existing variable name differs, inspect the `.env` variable names without displaying secret values and use the existing correct name.

Create `.env.example` only with placeholder values, never real credentials.

### 9.5 `.gitignore`

Include at least:

```gitignore
.venv/
.env
__pycache__/
*.py[cod]
chroma_db/
.gradio/
```

Do not ignore the source PDF unless the project owner requests it. Do not commit secrets.

---

## 10. Application implementation guidance

### 10.1 Startup and indexing logic

At application startup:

1. Load environment variables.
2. Validate that the Groq API key exists.
3. Validate that the PDF exists.
4. Initialize the embedding model.
5. Check whether the persistent ChromaDB index already exists and contains documents.
6. If it exists, load it.
7. Otherwise:
   - Load the PDF with `PyPDFLoader`.
   - Split documents with `RecursiveCharacterTextSplitter`.
   - Create the ChromaDB vector store using the selected embeddings.
   - Persist the index.
8. Create the retriever with `search_type="similarity"` and `search_kwargs={"k": 5}`.
9. Initialize the Groq chat model.
10. Launch the Gradio interface.

Avoid re-embedding the entire PDF on every application restart.

If a reliable index check is not available through the selected API, use a simple local marker or a documented rebuild approach. Do not add a separate database just to track ingestion.

### 10.2 Question-answering logic

For every submitted question:

1. Validate that the input is not blank.
2. Call the retriever to obtain five relevant documents.
3. Extract the text and page metadata from those documents.
4. Format the retrieved context.
5. Apply the RAG prompt.
6. Invoke the Groq LLM.
7. Return the answer to Gradio.
8. Add the question and answer to the session's conversation history.

Keep the data flow easy to follow. A small `answer_question()` function is acceptable, but do not create classes or a complex chain architecture.

### 10.3 Context formatting

For each retrieved chunk, include its page number if available.

Example:

```text
[Page 12]
Retrieved passage text...

[Page 18]
Retrieved passage text...
```

This makes it easier for the model to refer to the source pages.

### 10.4 Gradio history behavior

Use Gradio session state or a simple session-scoped state object so separate browser sessions do not accidentally share one user's history.

For this POC:
- History only needs to live while the application/session is active.
- It does not need to survive a restart.
- Do not persist history in ChromaDB.
- Do not create a separate history database.

Use the current Gradio API for chat message formatting. Confirm that the chosen `Chatbot` message format and event callback signatures match the installed Gradio version.

---

## 11. Non-functional requirements

| Category | Requirement |
|---|---|
| Simplicity | Beginner-friendly, procedural Python |
| Maintainability | Small number of files and minimal abstractions |
| Performance | Suitable for a small PDF and a local CPU embedding model |
| Retrieval | Similarity search, top 5 chunks |
| Persistence | Local ChromaDB index persists across restarts |
| Security | API key read from `.env`; never hard-coded |
| Privacy | No external storage of questions or answers |
| Reliability | Clear errors for missing files, keys, and API failures |
| Explainability | Show page references where available |
| Portability | Runs locally on a standard Python environment |
| Hardware | No GPU required for embeddings |
| UI | Simple Gradio interface with session history |

The LLM inference itself is performed through Groq's hosted API, so an internet connection is required for answer generation.

---

## 12. Smoke testing and acceptance criteria

Perform at least the following two smoke tests after the application is implemented and the PDF has been indexed.

### Smoke test 1 — Direct factual lookup

**Purpose:** Verify that the application can retrieve a specific fact from the PDF.

- Choose a specific error, symptom, command, fault condition, or troubleshooting instruction that genuinely appears in the supplied PDF.
- Ask a direct question about it.
- Confirm that the answer is consistent with the retrieved document passage.
- Confirm that the answer includes a page reference when available.
- Confirm that the application does not invent details.

**Acceptance criteria:**
- The question is accepted by Gradio.
- The retriever returns up to five relevant chunks.
- The LLM returns a relevant factual answer.
- The answer is grounded in the retrieved context.
- The UI updates the chat and left-side history.

### Smoke test 2 — Broader conceptual question

**Purpose:** Verify that the application can synthesize information from multiple relevant passages.

Suggested question:

> Based on the Cisco Nexus 9000 troubleshooting guide, describe a sensible step-by-step approach an engineer can follow to investigate a recurring network or switch issue. Explain how the engineer can use the documented symptoms and troubleshooting procedures to narrow down the possible cause.

**Acceptance criteria:**
- The application retrieves up to five relevant chunks.
- The answer synthesizes relevant information from the retrieved context.
- The answer is structured and understandable.
- Page references are included when available.
- Unsupported details are not presented as facts from the guide.
- The UI updates the chat and left-side history.

### Additional basic checks

- Start the application with a valid `.env` and confirm it initializes.
- Confirm the ChromaDB index is created on the first run.
- Restart the app and confirm that the existing index is reused.
- Submit an empty question and confirm that it is handled gracefully.
- Temporarily test a missing PDF path and confirm that a useful error is shown.
- Confirm that `sys.executable` points to `.venv`.
- Confirm that dependencies were installed inside `.venv`.

### Test reporting

At completion, provide a short test report with:
- Test name.
- Question used.
- Whether the test was actually executed.
- Pass/fail status.
- Brief observation about the answer and retrieval.
- Any limitation or blocker.

Do not mark a test as passed merely because the code was written. If the PDF or API credentials are unavailable, state exactly which test could not be completed and why.

---

## 13. README and run instructions

Create a concise `README.md` containing:

- Project overview.
- Prerequisites.
- Expected project structure.
- Virtual environment creation and activation commands.
- Dependency installation command.
- `.env` configuration instructions.
- Where to place the Cisco PDF.
- How to run the application.
- How to rebuild the ChromaDB index after replacing the PDF.
- How to run the two smoke tests.
- Troubleshooting tips for common setup errors.

Expected Windows PowerShell run sequence:

```powershell
# Navigate to the project directory
cd path\to\cisco-nexus-rag

# Create the environment if it does not already exist
py -3.12 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Verify interpreter location
python -c "import sys; print(sys.executable)"

# Start the application
python app.py
```

If `.venv` already exists, do not recreate it unnecessarily. If the user's shell is not PowerShell, provide equivalent activation instructions.

---

## 14. Suggested implementation sequence for Copilot

Implement in this order, verifying each stage before moving on:

1. Inspect the project directory and existing `.env` file. Do not reveal secret values.
2. Identify the available compatible Python version.
3. Create `.venv` in the project root.
4. Create the project folders and `.gitignore`.
5. Create `requirements.txt`.
6. Activate `.venv` and install all dependencies into it.
7. Verify the interpreter and pip paths are inside `.venv`.
8. Implement PDF loading.
9. Implement recursive text splitting with chunk size 512 and overlap 60.
10. Implement local `all-MiniLM-L6-v2` embeddings.
11. Implement persistent ChromaDB indexing and index reuse.
12. Implement the top-five similarity retriever.
13. Implement Groq GPT-OSS 120B initialization using the existing environment key.
14. Implement the grounded RAG prompt and answer generation.
15. Implement the Gradio two-column chat and history interface.
16. Create `README.md`.
17. Run syntax/import checks and start the application.
18. Execute the factual and conceptual smoke tests if the PDF and API are available.
19. Fix issues discovered during testing.
20. Provide a concise completion summary, exact run command, environment verification, and honest test results.

---

## 15. Definition of done

The project is complete when all applicable items below are satisfied:

- [ ] Project-local `.venv` exists.
- [ ] All dependencies are installed inside `.venv`.
- [ ] No packages were installed into global Python.
- [ ] Existing `.env` file is preserved.
- [ ] Groq API key is read securely from the environment.
- [ ] GPT-OSS 120B is configured through Groq using a supported model identifier.
- [ ] Cisco Nexus 9000 PDF is loaded successfully.
- [ ] PDF text is split using recursive character splitting (512 chunk size, 60 overlap).
- [ ] Local `all-MiniLM-L6-v2` embeddings are used.
- [ ] ChromaDB index is persisted locally.
- [ ] Existing index is reused on restart.
- [ ] Retriever uses similarity search with `k=5`.
- [ ] Retrieved context is passed to the LLM.
- [ ] Answers are grounded in retrieved passages.
- [ ] Unsupported answers are handled transparently.
- [ ] Page references are displayed where available.
- [ ] Gradio chat UI works.
- [ ] Left-side question-and-answer history updates correctly.
- [ ] README includes setup and run instructions.
- [ ] Factual smoke test is executed or its blocker is documented.
- [ ] Conceptual smoke test is executed or its blocker is documented.
- [ ] Final implementation summary and test results are provided.

---

## 16. Final guidance

This is deliberately a **basic RAG POC**. The goal is to demonstrate the core RAG lifecycle clearly:

**PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Top-5 Retrieval → Prompt + Context → Groq GPT-OSS 120B → Gradio Answer.**

Favor clarity over sophistication. A learner should be able to open `app.py` and understand the entire workflow without navigating a large codebase.

Do not over-engineer the application. Do not add classes, complex abstractions, additional databases, agentic workflows, reranking, or unrelated features. Deliver a small, reliable, demonstrable application that follows this PRD.
