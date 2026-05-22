"""
Terminal Chat Client — Personal Digital Assistant

This is the chat interface for the RAG pipeline. It is provided
complete — you do not need to modify this file.

Usage:
    python chat.py
"""

import os
import subprocess

from rag import Assistant

from dotenv import load_dotenv

WELCOME = """
╔══════════════════════════════════════════════════════╗
║           Personal Digital Assistant                 ║
║                                                      ║
║  Ask me about emails, notes, SMS, and calendar.      ║
║  Type '/clear' to reset conversation history.        ║
║  Filter by type: /calendar /email /notes /sms       ║
║  Try: search in my /calendar for dentist appointment ║
║  Type '/exit' to leave.                              ║
╚══════════════════════════════════════════════════════╝
"""


def load_config_from_env() -> dict[str, str | None]:
    """Load raw RAG configuration values from environment variables."""
    return {
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_API_URL"),
        "model": os.getenv("LLM_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "top_k": os.getenv("TOP_K"),
        "chunk_size": os.getenv("CHUNK_SIZE"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP"),
        "overfetch_multiplier": os.getenv("OVERFETCH_MULTIPLIER"),
    }


def main():
    print("Initializing assistant...")
    config = load_config_from_env()
    assistant = Assistant.from_config(config)
    subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)

    print(WELCOME)

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() == "/exit":
            print("Goodbye!")
            break

        if question.lower() == "/clear":
            assistant.clear_history()
            subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)
            print("\nConversation history cleared.\n")
            continue

        response = assistant.ask(question)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    load_dotenv()
    main()
