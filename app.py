import argparse
import logging
import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT_DIR = Path(__file__).resolve().parent
PDF_PATH = ROOT_DIR / "data" / "cisco_nexus_9000_troubleshooting_guide.pdf"
CHROMA_PATH = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "cisco_nexus_troubleshooting"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "openai/gpt-oss-120b"
RETRIEVER_K = 5

SYSTEM_PROMPT = """You are a technical assistant specializing in the Cisco Nexus 9000 Series Troubleshooting Guide.
Answer using only the supplied retrieved document context as your primary and authoritative source.
Do not invent technical facts, commands, configuration steps, error messages, or procedures.
Every technical assertion must be supported by the retrieved context. Synthesize only by combining facts explicitly stated there; do not infer unmentioned causes, counter values, or command effects.
When explaining a procedure, state what the guide says to check without inferring what the result proves. If the context does not contain enough information for part of the question, say so rather than filling the gap.
Do not interpret a log entry or link state beyond the meaning explicitly stated in the context, and do not infer a root cause from an example.
Keep possible causes separate from recommended actions; do not turn a stated possibility into an instruction unless the guide explicitly recommends that action.
Clearly label any explanation that goes beyond an explicitly stated fact.
Be concise, structured, and technically clear. Cite the context's page number for each relevant claim when available.

Retrieved document context:
{context}

User question:
{question}

Answer:"""


def answer_question(question, retriever, llm):
    question = question.strip()
    if not question:
        raise ValueError("Enter a question before submitting.")

    documents = retriever.invoke(question)
    if not documents:
        return "The supplied document context does not provide sufficient information to answer this question.", []

    context_parts = []
    for document in documents[:RETRIEVER_K]:
        page = document.metadata.get("page")
        if isinstance(page, int):
            context_parts.append(f"[Page {page + 1}]\n{document.page_content}")
        else:
            context_parts.append(document.page_content)

    prompt = SYSTEM_PROMPT.format(
        context="\n\n".join(context_parts),
        question=question,
    )
    response = llm.invoke(prompt)
    return response.content, documents[:RETRIEVER_K]


def initialize_rag():
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to the project .env file.")
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Troubleshooting PDF not found at {PDF_PATH}. "
            "Place the guide at data/cisco_nexus_9000_troubleshooting_guide.pdf."
        )

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    except Exception as error:
        logging.exception("Could not initialize the local embedding model.")
        raise RuntimeError(
            "Could not initialize the local embedding model. Check the network connection "
            "and available disk space, then restart the application."
        ) from error

    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PATH),
        )
        if vectorstore._collection.count() == 0:
            pages = PyPDFLoader(str(PDF_PATH)).load()
            if not any(page.page_content.strip() for page in pages):
                raise ValueError(
                    "The supplied PDF contains no extractable text. Scanned PDFs requiring OCR are not supported."
                )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=60,
                add_start_index=True,
            )
            chunks = splitter.split_documents(pages)
            if not chunks:
                raise ValueError("No searchable text chunks could be created from the supplied PDF.")
            vectorstore.add_documents(chunks)
            logging.info("Indexed %s chunks from the troubleshooting guide.", len(chunks))
        else:
            logging.info("Reusing the existing local ChromaDB index.")
    except Exception as error:
        logging.exception("Could not load or index the troubleshooting guide.")
        if isinstance(error, (FileNotFoundError, ValueError)):
            raise
        raise RuntimeError(
            "Could not initialize or retrieve the local document index. "
            "Check the PDF and chroma_db folder, then see the terminal log."
        ) from error

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    try:
        llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0.2,
            api_key=api_key,
        )
    except Exception as error:
        logging.exception("Could not initialize the Groq chat model.")
        raise RuntimeError(
            "Could not initialize the Groq model. Verify the configured model and API key."
        ) from error

    return retriever, llm


def render_history(entries):
    messages = []
    for entry in entries[-10:]:
        messages.extend(
            [
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": entry["answer"]},
            ]
        )
    return messages


def make_submit_handler(retriever, llm):
    def submit_question(question, chat_messages, session_history):
        if not question or not question.strip():
            raise gr.Error("Enter a question before submitting.")

        try:
            answer, _ = answer_question(question, retriever, llm)
        except Exception as error:
            logging.exception("Question answering failed.")
            raise gr.Error(
                "I could not answer this question. Check the terminal log and verify "
                "your Groq API connection and configuration."
            ) from error

        chat_messages = list(chat_messages or [])
        session_history = list(session_history or [])
        clean_question = question.strip()
        chat_messages.extend(
            [
                {"role": "user", "content": clean_question},
                {"role": "assistant", "content": answer},
            ]
        )
        session_history.append({"question": clean_question, "answer": answer})
        return chat_messages, render_history(session_history), session_history, ""

    return submit_question


def build_app(retriever, llm):
    submit_question = make_submit_handler(retriever, llm)

    with gr.Blocks(title="Cisco Nexus 9000 Troubleshooting Assistant") as demo:
        gr.Markdown(
            "# Cisco Nexus 9000 Troubleshooting Assistant\n"
            "Answers are generated from the supplied Cisco Nexus 9000 troubleshooting guide."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("## Conversation History")
                history_view = gr.Chatbot(
                    type="messages",
                    height=600,
                    show_label=False,
                    allow_tags=False,
                )
                clear_button = gr.Button("Clear History", size="sm")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=600,
                    label="Chat",
                    allow_tags=False,
                )
                with gr.Row():
                    question_box = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about the troubleshooting guide...",
                        lines=2,
                        scale=5,
                    )
                    ask_button = gr.Button("Ask", variant="primary", scale=1)
                clear_chat_button = gr.Button("Clear Chat", size="sm")

        session_history = gr.State([])
        outputs = [chatbot, history_view, session_history, question_box]
        ask_button.click(
            submit_question,
            inputs=[question_box, chatbot, session_history],
            outputs=outputs,
        )
        question_box.submit(
            submit_question,
            inputs=[question_box, chatbot, session_history],
            outputs=outputs,
        )
        clear_button.click(
            lambda: ([], [], []),
            outputs=[chatbot, history_view, session_history],
        )
        clear_chat_button.click(lambda: [], outputs=chatbot)

    return demo


def run_smoke_tests(retriever, llm):
    questions = [
        (
            "What troubleshooting actions does the guide recommend for a port that remains "
            "in a Link Failure or Not Connected state?"
        ),
        (
            "Compare the guide's advice for (1) a port that remains in Link Failure "
            "or Not Connected state and (2) unexpected link flapping. Summarize the "
            "documented checks and exact facts stated about link states. If the "
            "passages do not say how to interpret something or identify who initiated "
            "a flap, explicitly say so."
        ),
    ]
    for label, question in zip(("Direct factual lookup", "Broader conceptual question"), questions):
        answer, documents = answer_question(question, retriever, llm)
        pages = sorted(
            {
                document.metadata["page"] + 1
                for document in documents
                if isinstance(document.metadata.get("page"), int)
            }
        )
        print(f"\n{label}\nQuestion: {question}\nRetrieved chunks: {len(documents)}")
        print(f"Retrieved pages: {pages or 'not available'}\nAnswer:\n{answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Cisco Nexus 9000 troubleshooting RAG demo")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the direct factual and broader conceptual questions instead of launching Gradio.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    retriever, llm = initialize_rag()
    if args.smoke_test:
        sys.stdout.reconfigure(encoding="utf-8")
        run_smoke_tests(retriever, llm)
    else:
        build_app(retriever, llm).launch()


if __name__ == "__main__":
    main()
