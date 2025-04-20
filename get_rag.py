from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os, json


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI()

chroma_client = chromadb.HttpClient("localhost", 8010)
openai_ef = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_collection("monopoly-guide", embedding_function=openai_ef)

result = collection.query(
    query_texts=["What is player role of banker?"],
    n_results=3,
    include=["distances", "documents"]
)

# print(json.dumps(result, indent=3, sort_keys=True))

response = openai_client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = [
        {
            "role": "system",
            "content": f"Your job is to answer based on provided context only, This is the context {result.get("documents")}"
        },
        {
            "role": "user",
            "content": "What is player role of banker?"
        }
    ]
)

content = response.choices[0].message.content
print(content)
