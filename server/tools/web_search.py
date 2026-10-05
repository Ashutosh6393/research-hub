import os

from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs, and Snippets."""


    results = tavily.search(query, max_results=5)

    out = []

    for r in results['results']:
        out.append(f'"Title": {r['title']}, \n"URL": {r['url']}, \n"Snippet": {r['content'][:300]}"')

    print(out)

    return "\n----\n.".join(out)




