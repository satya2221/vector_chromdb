from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

with open("monopoly.txt", "r") as f:
    text = f.read()
    
splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))

documents = splitter.create_documents([text])

print(documents)