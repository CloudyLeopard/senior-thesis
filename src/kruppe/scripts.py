
import asyncio
import os
import json
import argparse
from datetime import datetime

from kruppe.scraper import NewYorkTimesData, FinancialTimesData, NewsAPIData
from kruppe.document_store import AsyncMongoDBStore

async def scrape_news_feed(nyt=True, ft=True):
    """this scrapes approximately a month worth of news articles"""
    today_date = datetime.now().date().isoformat()
    document_store = await AsyncMongoDBStore.create(uri=os.getenv("MONGODB_URI"), db_name="FinancialNews", collection_name=f"news_feed_{today_date}")

    if nyt:
        with open("./.nyt-headers.json") as f:
            nyt_headers = json.load(f)
        nyt_source = NewYorkTimesData(headers=nyt_headers)

        print("Scraping New York Times...")
        nyt_documents = await nyt_source.fetch_news_feed(num_results=250)

        print(f"Saving New York Times... ({len(nyt_documents)} documents)")
        await document_store.save_documents(nyt_documents)

    if ft:
        with open("./.ft-headers.json") as f:
            ft_headers = json.load(f)
        ft_source = FinancialTimesData(headers=ft_headers)

        print("Scraping Financial Times...")
        links = await ft_source.fetch_news_feed(days=30)
        ft_documents = await ft_source.async_scrape_links(links)

        print(f"Saving Financial Times... ({len(ft_documents)} documents)")
        await document_store.save_documents(ft_documents)

async def scrape_news_search(query, nyt=True, ft=True, newsapi=True, num_results=30):
    today_date = datetime.now().date().isoformat()
    document_store = await AsyncMongoDBStore.create(uri=os.getenv("MONGODB_URI"), db_name="FinancialNews", collection_name=f"news_search_{today_date}")

    if nyt:
        with open("./.nyt-headers.json") as f:
            nyt_headers = json.load(f)
        nyt_source = NewYorkTimesData(headers=nyt_headers)

        print("Scraping New York Times...")
        nyt_documents = await nyt_source.async_fetch(query, num_results=num_results, sort="newest")

        print(f"Saving New York Times... ({len(nyt_documents)} documents)")
        await document_store.save_documents(nyt_documents)

    if ft:
        with open("./.ft-headers.json") as f:
            ft_headers = json.load(f)
        ft_source = FinancialTimesData(headers=ft_headers)

        print("Scraping Financial Times...")
        ft_documents = await ft_source.async_fetch(query, num_results=num_results, sort="date")

        print(f"Saving Financial Times... ({len(ft_documents)} documents)")
        await document_store.save_documents(ft_documents)
    
    if newsapi:
        newsapi_source = NewsAPIData()

        print("Scraping NewsAPI...")
        newsapi_documents = await newsapi_source.async_fetch(query, num_results=num_results, sort_by="publishedAt")

        print(f"Saving NewsAPI... ({len(newsapi_documents)} documents)")
        await document_store.save_documents(newsapi_documents)

def main():
    parser = argparse.ArgumentParser(description="Scrape news feed and save to MongoDB.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', action='store_true', help="Call function for searching news")
    group.add_argument('-n', action='store_true', help="Call function for scraping news feed")

    parser.add_argument('--nyt', action='store_true', help="Scrape today's NYT newswire")
    parser.add_argument('--ft', action='store_true', help="Scrape today's FT newsfeed")
    parser.add_argument('--newsapi', action='store_true', help="Scrape NewsAPI")
    parser.add_argument('--all', action='store_true', help="Scrape all newsfeeds")

    args = parser.parse_args()

    if args.s:
        query = input("Enter query: ")
        asyncio.run(scrape_news_search(query, nyt=args.nyt or args.all, ft=args.ft or args.all, newsapi=args.newsapi or args.all))

    if args.n:
        asyncio.run(scrape_news_feed(nyt=args.nyt or args.all, ft=args.ft or args.all))
    

if __name__ == "__main__":
    # main_load()
    main()