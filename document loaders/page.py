from langchain_community.document_loaders import WebBaseLoader

url = "https://python.langchain.com/docs/introduction/"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content[:1000])  # first 1000 chars