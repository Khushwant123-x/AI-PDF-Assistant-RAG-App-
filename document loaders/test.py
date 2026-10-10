from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader(
    r"C:\Users\Jyoti\OneDrive\Desktop\RAG Project\document loaders\sample.txt"
)

docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,      # 10 characters per chunk
    chunk_overlap=1     # 1 character overlap
)

chunks = text_splitter.split_documents(docs)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk.page_content)
    print()