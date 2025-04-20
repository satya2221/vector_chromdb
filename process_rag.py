from mistralai import Mistral
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Mistral(
    api_key=MISTRAL_API_KEY
)

chroma_client = chromadb.HttpClient("localhost", 8010)

openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

ocr_response = client.ocr.process(
    model = "mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://www.hasbro.com/common/instruct/00009.pdf"
    }
)
content = ocr_response.model_dump() # di convert ke dictionary

markdown_content = ""

for page in content.get("pages", []):
    if "markdown" in page:
        markdown_content += page["markdown"] + "\n\n"

splitter = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-small"))
documents = splitter.create_documents([markdown_content])

collection = chroma_client.create_collection(
    "monopoly-guide",
    embedding_function=openai_ef
)
collection.add(
    documents=[doc.model_dump().get("page_content")  for doc in documents],
    ids=[str(i) for i in range(len(documents))]
)