from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("monopoly.txt", "r") as f:
    text = f.read()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 1 chunk ada 1000 character
    chunk_overlap=200 # misal ada konteks yang gak harusnya kebawa, maka ia akan mundur 200 char ke belakang  
)

documents = splitter.create_documents([text])

print(documents)