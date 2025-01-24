
import asyncio
import os
import json
import argparse
from datetime import datetime

from rag.scraper import NewYorkTimesData, FinancialTimesData
from rag.document_store import AsyncMongoDBStore

async def scrape_news_feed(nyt=True, ft=True):
    today_date = datetime.now().date().isoformat()
    document_store = await AsyncMongoDBStore.create(uri=os.getenv("MONGODB_URI"), db_name="FinancialNews", collection_name=f"news_{today_date}")

    if nyt:
        with open("./.nyt-headers.json") as f:
            nyt_headers = json.load(f)
        nyt_source = NewYorkTimesData(headers=nyt_headers)

        print("Scraping New York Times...")
        nyt_documents = await nyt_source.fetch_news_feed(num_results=100)

        print(f"Saving New York Times... ({len(nyt_documents)} documents)")
        await document_store.save_documents(nyt_documents)

    if ft:
        with open("./.ft-headers.json") as f:
            ft_headers = json.load(f)
        ft_source = FinancialTimesData(headers=ft_headers)

        print("Scraping Financial Times...")
        links = await ft_source.fetch_news_feed(days=1)
        ft_documents = await ft_source.async_scrape_links(links)

        print(f"Saving Financial Times... ({len(ft_documents)} documents)")
        await document_store.save_documents(ft_documents)


def main():
    parser = argparse.ArgumentParser(description="Scrape news feed and save to MongoDB.")
    parser.add_argument('--nyt', action='store_true', help="Scrape today's NYT newswire")
    parser.add_argument('--ft', action='store_true', help="Scrape today's' FT newsfeed")
    parser.add_argument('--all', action='store_true', help="Scrape all newsfeeds")

    args = parser.parse_args()

    asyncio.run(scrape_news_feed(nyt=args.nyt or args.all, ft=args.ft or args.all))
    

if __name__ == "__main__":
    # main_load()
    main()