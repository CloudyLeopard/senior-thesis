import requests
from typing import List
import logging
from pydantic import Field, model_validator
from typing import Any

from lexisnexisapi import credentials, webservices

from kruppe.scraper.base_source import BaseDataSource, RequestSourceException
from kruppe.scraper.utils import WebScraper
from kruppe.models import Document

logger = logging.getLogger(__name__)

class LexisNexisData(BaseDataSource):
    source: str = Field(default="LexisNexis")
    token: str = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_credentials_and_set_token(cls, data: Any):
        if isinstance(data, dict):
            if data.get("token") is not None:
                return data
            else:
                # Fetch credentials
                cred = credentials.get_Credentials()
                client_id = cred.get("WSAPI_CLIENT_ID")
                secret = cred.get("WSAPI_SECRET")

                if not client_id or not secret:
                    logger.error(
                        "LexisNexis credentials not set. Please set them in %s",
                        credentials.cred_file_path(),
                    )
                    raise ValueError("LexisNexis credentials not set")

                # Generate and set the token
                data["token"] = webservices.token()
                return data
        return data

    def fetch(self, query: str, num_results=10, **kwargs) -> List[Document]:
        """
        Fetch documents from Lexis Nexis based on query

        see https://dev.lexisnexis.com/dev-portal/documentation/News#/News%20API/get_News for documentation

        Args:
            query: query to retrieve text from
            num_results: number of results to retrieve (default: 10)

        Returns:
            List of Document objects with text and metadata

        Raises:
            ValueError: if response is invalid
            RuntimeError: if Lexis Nexis query limit is reached
        """
        search_string = query  # TODO: adjust this

        # TODO: adjust parameter based on documentation
        parameters = {
            "$search": search_string,
            "$expand": "Document",  # "Document" to get html data
            "$top": str(
                num_results
            ),  # Sets the maximum number of results to receive for this request.
            # Filter with two conditions
            "$filter": "Language eq LexisNexis.ServicesApi.Language'English' and year(Date) eq 2024",
            "$select": "ResultId, Title, Source",
        }

        try:
            logger.debug("Fetching documents from Lexis Nexis API")
            data = webservices.call_api(
                access_token=self.token, endpoint="News", params=parameters
            )
        except requests.exceptions.HTTPError as e:
            msg = e.response.reason
            if e.response.status_code == 429:
                msg = "Lexis Nexis query limit reached"
            logger.error("Lexis Nexis HTTP Error %d: %s", e.response.status_code, msg)
            raise RequestSourceException(msg)
        except requests.exceptions.RequestException as e:
            logger.error("Lexis Nexis Failed to fetch documents: %s", e)
            raise RequestSourceException(e)

        logger.debug("Converting data to Document objects")
        documents = []
        for result in data["value"]:
            html = result["Document"]["Content"]
            try:
                data = WebScraper.default_html_parser(html)
                text = data["content"]
            except Exception:
                text = html  # fallback to html if scraping fails

            metadata = self.parse_metadata(
                query=query,
                title=result["Title"],
                source=result["Source"]["Name"],
                lexisResultId=result["ResultId"],
                citation=result["Document"].get("Citation", ""),
            )

            document = Document(text=text, metadata=metadata)

            documents.append(document)

        logger.debug(
            "Successfully fetched %d documents from Lexis Nexis API", len(documents)
        )
        return documents

    async def async_fetch(self, query: str, num_results=10, **kwargs) -> List[Document]:
        """Fallback to sync fetch"""
        return self.fetch(query, num_results)

