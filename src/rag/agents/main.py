import argparse
from datetime import datetime
import json
import logging
import asyncio

from rag.document_store import MongoDBStore
from rag.tools.searcher import NewsArticleSearcher
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


async def async_load_data(query: str, num_results: int, verbose: bool = False):
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
    documents = await searcher.async_search(query, num_results=num_results, months=12)

    logger.debug("Finished scraping data for query: %s", query)
    logger.debug("Saving %d documents to document store", len(documents))
    document_store.save_documents(documents)

    logger.debug("Finished saving documents to document store")

async def async_export_data(query: str, num_results: int, verbose: bool = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.DEBUG)

    with open(".ft-headers.json") as f:
        ft_headers = json.load(f)
    sources = [
        LexisNexisData(),
        FinancialTimesData(headers=ft_headers),
        NewsAPIData(),
    ]

    logging.info("Initialized %d data sources", len(sources))

    logger.debug("Begin scraping data for query: %s", query)
    searcher = NewsArticleSearcher(sources=sources)
    # TODO make "months" a parameter (or some other form like "from" or whatever)
    documents = await searcher.async_search(query, num_results=num_results, months=6)

    logger.info("Finished scraping data for query \"%s\". Obtained %d documents", query, len(documents))

    today = datetime.now().date().isoformat()
    path = f"./exports/{query}/{today}"
    logger.debug("Exporting document into path %s", path)
    searcher.export_documents(path, create_dir=True)

def main():
    parser = argparse.ArgumentParser(description="Load data for RAG.")
    parser.add_argument("query", type=str, help="The query for sources to retrieve data for.")
    parser.add_argument("-e", "--export", action="store_true", help="Export the data to a directory instead of document storage.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--num_results", type=int, default=10, help="10 results per source")
    
    args = parser.parse_args()

    if args.export:
        asyncio.run(async_export_data(args.query, num_results=args.num_results, verbose=args.verbose))
        return
    
    asyncio.run(async_load_data(args.query, num_results=args.num_results, verbose=args.verbose))

if __name__ == "__main__":
    asyncio.run(async_load_data("AAPL", verbose=True))