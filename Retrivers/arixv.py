import arxiv

search = arxiv.Search(
    query="large language models",
    max_results=3
)

for result in search.results():
    print(result.title)