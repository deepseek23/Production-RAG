from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import TextLoader

load_dotenv()

def load_text():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b'heelo this , is the same file. \n my name is tarun.\n i am studying from freecodecamp.')
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text()