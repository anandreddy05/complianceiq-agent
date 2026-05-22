from langchain_community.document_loaders import WebBaseLoader


def scrape_url(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text[:3000]
