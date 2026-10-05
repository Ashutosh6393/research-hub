import requests
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text ccontent from a given URL for deeper reading"""

    try:
        resp = requests.get(url, timeout=10, headers={"User-agent": "Mozilla/5.0 "})

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
        
        return soup.get_text(separator=" ", strip=True)[:3000]

    except requests.exceptions.RequestException as e:
        return f"Error scraping URL {url}: {e}"