import argparse
from datetime import datetime
import json
import logging
import asyncio

from rag.document_storages import MongoDBStore
from rag.agents.searcher import NewsArticleSearcher
from rag.tools.sources import (
    LexisNexisData,
    FinancialTimesData,
    NewsAPIData,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
logger.addHandler(handler)


async def async_load_data(query: str, verbose: bool = False):
    """given query, scrape relevant news articles, store in document store and vector store"""
    if verbose:
        logger.setLevel(logging.DEBUG)

    # loading in all the api key thru environment variables
    today = datetime.now().date().isoformat()
    document_store = MongoDBStore(collection_name=today)

    with open(".ft-headers.json") as f:
        ft_headers = json.load(f)
    sources = [
        LexisNexisData(),
        FinancialTimesData(headers=ft_headers),
        NewsAPIData(),
    ]

    logger.debug("Begin scraping data for query: %s", query)
    searcher = NewsArticleSearcher(sources=sources)
    documents = await searcher.async_search(query)

    logger.debug("Finished scraping data for query: %s", query)
    logger.debug("Saving %d documents to document store", len(documents))
    document_store.save_documents(documents)

    logger.debug("Finished saving documents to document store")

def main():
    parser = argparse.ArgumentParser(description="Load data for RAG.")
    parser.add_argument("query", type=str, help="The query for sources to retrieve data for.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()

    asyncio.run(async_load_data(args.query, verbose=args.verbose))

if __name__ == "__main__":
    asyncio.run(async_load_data("AAPL", verbose=True))