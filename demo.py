from dotenv import load_dotenv
from langsmith import traceable
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


# =========================
# Local Ollama LLM
# =========================

llm = ChatOllama(
    model="llama3.1:8b",   # change this to your Ollama model
    temperature=0
)


# =========================
# 1. Format Prompt
# =========================

@traceable
def format_prompt(subject):
    messages = [
        SystemMessage(
            content="You are a helpful AI assistant. "
                    "Explain concepts clearly and concisely."
        ),
        HumanMessage(
            content=f"Explain the following topic in simple terms:\n\n{subject}"
        )
    ]

    return messages


# =========================
# 2. Invoke LLM
# =========================

@traceable(run_type="llm")
def invoke_llm(messages):
    response = llm.invoke(messages)

    return response


# =========================
# 3. Parse Output
# =========================

@traceable
def parse_output(response):
    return response.content.strip()


# =========================
# 4. Complete Pipeline
# =========================

@traceable
def run_pipeline():
    messages = format_prompt("Retrieval Augmented Generation")

    response = invoke_llm(messages)

    answer = parse_output(response)

    return answer


# =========================
# Run
# =========================

if __name__ == "__main__":
    result = run_pipeline()

    print("\nFinal Answer:\n")
    print(result)