from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain.chat_models import init_chat_model



print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():

    # Test openai
    llm = init_chat_model('groq:llama-3.1-8b-instant', temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from groq: {response}")

    # Test anthropic
    llm_anthropic = init_chat_model('google_genai:gemini-3.5-flash-lite', temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from google: {response_anthropic}")

    print("Setup complete!")


if __name__ == "__main__":
    main()