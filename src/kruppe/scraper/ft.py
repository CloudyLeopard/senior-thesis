import httpx
from typing import AsyncGenerator, List, Dict
from bs4 import BeautifulSoup
import asyncio
import logging
from datetime import datetime, timedelta
import re

from kruppe.scraper.base_source import BaseDataSource, RequestSourceException
from kruppe.scraper.utils import WebScraper, HTTPX_CONNECTION_LIMITS
from kruppe.models import Document

logger = logging.getLogger(__name__)

# NOTE: for scraping FT, could be even easier to use
# https://www.ft.com/sitemaps/index.xml

class FinancialTimesData(BaseDataSource):
    headers: Dict[str, str]
    source: str = "Financial Times"

    @staticmethod
    def _parse_search_page(html: str) -> List[str]:
        # get the list of links from the search page's html
        soup = BeautifulSoup(html, "lxml")

        # get all search items links
        # NOTE: 1 page = 25 search results
        links = []
        search_divs = soup.find_all("div", class_="search-item")
        for div in search_divs:
            a_tag = div.find("a", class_="js-teaser-heading-link")
            if a_tag:
                links.append(a_tag.get("href"))

        return links

    @staticmethod
    def _ft_blog_html_parser(html: str, url: str, post_id: str) -> Dict[str, str]:
        """FT has 'blogs', which are a different format than FT's 'articles'.
        Often, multiple blog posts with very different topic can be on the same
        page. So, we use "post_id" to isolate the one we want, and this post_id
        can be found after the # in the returned urls from search page

        Returns:
            Dict[str, str]: A dictionary containing the content and other metadata of the blog post.
        """
        soup = BeautifulSoup(html, "lxml")
        if "post" not in post_id:
            post_id = f"post-{post_id}"
        post_id_element = soup.find(id=post_id)
        if not post_id_element:
            # if cannot find post_id in website, return None
            return None

        title = post_id_element.find("h2").get_text()
        content = "\n".join([p.text.strip() for p in post_id_element.find_all("p")])
        posted_time = post_id_element.find("time").get("datetime")

        return {
            "content": content, 
            "meta": {
                "title": title,
                "url": url,
                "publication_time": posted_time
            }
        }

    async def async_fetch(
        self, query: str, num_results: int = 25, sort="relevance", months: int = None, **kwargs
    ) -> AsyncGenerator[Document, None]:
        """Async version of fetch. Fetches links from Financial Times, scrapes them, and returns as a list of Documents.
        If document store is set, save documents to document store.

        Args:
            query (str): The main search query to fetch relevant links.
            num_results(int, optional): The number of search results to scrape. Defaults to 25 (or 1 page)
            sort (str, optional): The sort order of the search results. Accepted values are "date" and "relevance". Defaults to "relevance".

        Returns:
            List[Document]: A list of Document objects containing the text and metadata
                            of the scraped links.

        Raises:
            HTTPError: If the request to the Financial Times API fails.
        """
        links = []
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        try:
            # Search FT using query and scrape list of articles
            url = "https://www.ft.com/search"

            search_requests = []
            pages = num_results // 25 + 1
            for page in range(1, pages + 1):
                params = {
                    "q": query,
                    "sort": sort,  # date or relevance
                    "page": page,
                    "isFirstView": "false",
                }

                if months:
                    params["from"] = (datetime.now() - timedelta(days=months * 30)).date().isoformat()
                
                search_requests.append(client.build_request("GET", url, params=params))
            
            async def send_requests_for_links(search_request):
                try:
                    response = await client.send(search_request)
                    response.raise_for_status()  # TODO: error handling
                    html = response.text
                    links.extend(self._parse_search_page(html))
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Financial Times HTTP Error %d: %s", e.response.status_code, e.response.text
                    )
                    raise RequestSourceException(e.response.text)
                except httpx.RequestError as e:
                    logger.error("Error fetching Financial Times search page: %s", e)
                    raise RequestSourceException(e)
            
            await asyncio.gather(*map(send_requests_for_links, search_requests))

            logger.info("Fetched %d links from Financial Times on query %s", len(links), query)
            # FT articles has two types: regular articles that can be
            # scraped with default parser, and blogs where we just want to
            # extract the relevant blog portion
            article_links = []
            blog_links = []

            for link in links:
                if not link.startswith("https"):
                    link = f"https://www.ft.com{link}"
                if "#" in link:
                    blog_links.append(link)
                else:
                    article_links.append(link)

            # scrape articles
            logger.debug("Initialize Async WebScraper")
            scraper = WebScraper(async_client=client)

            logger.info("Async scraping %d Financial Times articles", len(links))
            async for data in scraper.async_scrape_links(article_links):
                if data is None:
                    continue

                metadata = self.parse_metadata(
                    query=query,
                    **data["meta"]
                )
                document = Document(text=data["content"], metadata=metadata)
                yield document

            
            logger.info("Async scraping %d Financial Times blogs", len(blog_links))
            # scrape blogs
            # NOTE: not using the default parser. The custom parser takes an additional input
            # so i am scraping the link individually, rather than scraping the whole list
            scraper.set_html_parser(self._ft_blog_html_parser)

            # NOTE: need to use asyncio.gather to scrape the blog links, since we can't call on
            # the ascrape_links method while using a custom html parser with custom input
            tasks = [
                scraper.async_scrape_link(url=link, post_id=link.split("#")[1])
                for link in blog_links
            ]
            for future in asyncio.as_completed(tasks):
                data = await future
                if data is None:
                    continue
                metadata = self.parse_metadata(
                    query=query,
                    **data["meta"]
                )
                yield Document(text=data["content"], metadata=metadata)
        finally:
            await client.aclose()
    
    async def fetch_news_feed(self, days: int = 365, num_results: int = None) -> List[str]:
        links = []
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        url = "https://www.ft.com/news-feed"
        pages = 1
        end = False

        def parse_datetime(datetime_str):
            # Check if the string ends with 'Z' (UTC format)
            if datetime_str.endswith("Z"):
                dt_format = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in datetime_str else "%Y-%m-%dT%H:%M:%SZ"
                dt_parsed = datetime.strptime(datetime_str, dt_format)
            else:
                # Handle timezone offset format
                # Regex to detect presence of milliseconds before timezone
                if re.search(r"\.\d+", datetime_str):
                    dt_format = "%Y-%m-%dT%H:%M:%S.%f%z"
                else:
                    dt_format = "%Y-%m-%dT%H:%M:%S%z"
                dt_parsed = datetime.strptime(datetime_str, dt_format)
            
            return dt_parsed

        if days:
            end_date = datetime.today().date()
            start_date = (datetime.today() - timedelta(days=days)).date()
            logger.info("Fetching news feed from %s to %s", start_date, end_date)

        try:
            while not end:
                logger.info("Current page: %d", pages)
                params = {
                    "page": pages,
                }

                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    html = response.text
                    soup = BeautifulSoup(html, "lxml")

                    a_tags = soup.find_all("a", class_="js-teaser-heading-link")
                    time_tags = soup.find_all("time")

                    logger.debug("Found %d articles on page %d", len(a_tags), pages)

                    for a_tag, time_tag in zip(a_tags, time_tags):
                        parsed_date = parse_datetime(time_tag['datetime']).date()
                        logger.debug("Article date: %s", parsed_date)

                        # filter by date
                        if days and parsed_date > end_date:
                            continue
                        if days and parsed_date < start_date:
                            end = True
                            break
                        
                        parsed_link = a_tag.get("href")
                        if not parsed_link.startswith("https"):
                            parsed_link = f"https://www.ft.com{parsed_link}"
                        links.append(parsed_link)
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Financial Times HTTP Status Error %d: %s", e.response.status_code, e.response.text
                    )
                    raise RequestSourceException(e.response.text)
                except httpx.RequestError as e:
                    logger.error("Financial Times Request Error: %s", e)
                    raise RequestSourceException(e)
                finally:
                    pages += 1
        finally:
            await client.aclose()

        if num_results:
            links = links[:num_results]
        
        logger.info("Fetched %d links from Financial Times news feed", len(links))
    
        return links
    
    async def async_scrape_links(self, links: List[str]):
        client = httpx.AsyncClient(timeout=10.0, headers=self.headers, limits=HTTPX_CONNECTION_LIMITS)
        try:
            # scrape articles
            logger.debug("Initialize Async WebScraper")
            scraper = WebScraper(async_client=client)

            logger.debug("Async scraping %d Financial Times articles", len(links))
            async for data in scraper.async_scrape_links(links):
                if data is None:
                    # if both selenium scrape and httpx scrape fails, article will return None
                    # in this case, we can't parse the metadata, so we skip
                    continue
                metadata = self.parse_metadata(
                    query=None,
                    **data["meta"]
                )
                document = Document(text=data["content"], metadata=metadata)
                yield document
        finally:
            await client.aclose()
        