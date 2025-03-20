from typing import List, Dict, AsyncGenerator, Generator
import pathlib
import logging
from typing import Optional
from pydantic import field_validator, model_validator

from kruppe.data_source.base_source import BaseDataSource
from kruppe.data_source.utils import WebScraper
from kruppe.models import Document

logger = logging.getLogger(__name__)

class DirectoryData(BaseDataSource):
    path: Optional[pathlib.Path] = None
    input_files: Optional[List[pathlib.Path]] = None
    source: str = "Local Directory"

    @field_validator("path", mode="after")
    @classmethod
    def validate_path(cls, v):
        if v is not None and not v.is_dir():
            raise ValueError("Invalid path - must be a directory")
        return v

    @field_validator("input_files", mode="after")
    @classmethod
    def validate_input_files(cls, v):
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("input_files must be a list")
            if not all(f.is_file() for f in v):
                raise ValueError("input_files must be valid file paths")
        return v

    @model_validator(mode="after")
    def validate_either_path_or_input_files(self):
        if self.path is None and self.input_files is None:
            raise ValueError("Either path or input_files must be set")
        return self

    def model_post_init(self, __context):
        if self.path is not None:
            self._file_generator = self.path.rglob("*")
        else:
            self._file_generator = self.input_files

    def fetch(self, query: str = None, **kwargs) -> Generator[Document, None, None]:
        """given path to data folder, fetch text files in 
        subdirectory with name matching query"""
        
        for file_path in self._file_generator:
            # if query is None, read everything
            # if query is not None, only read files with name matching query
            if query is not None: 
                if query not in file_path.name:
                    continue
            
            if file_path.suffix == ".txt":
                txt = file_path.read_text()

                metadata = self.parse_metadata(
                    query="NA",
                    name=file_path.name,
                    path=file_path.as_posix(),
                )
                document = Document(text=txt, metadata=metadata)
            if file_path.suffix == ".pdf":
                pdf_text, pdf_meta = self.simple_pdf_parser(file_path.as_posix())
                metadata = self.parse_metadata(
                    query="NA",
                    name=file_path.name,
                    path=file_path.as_posix(),
                    publication_time = pdf_meta.get("creation_date"),
                    title = pdf_meta.get("title")
                )
                document = Document(text=pdf_text, metadata=metadata)
            if file_path.suffix == ".html":
                html = file_path.read_text()
                scraped_data = WebScraper.default_html_parser(html)
                metadata = self.parse_metadata(
                    query="NA",
                    name=file_path.name,
                    path=file_path.as_posix(),
                    title=scraped_data["title"],
                    publication_time=scraped_data["time"],
                )
                document =Document(text=scraped_data["content"], metadata=metadata)
            yield document
    async def async_fetch(self, query: str = None, **kwargs) -> AsyncGenerator[Document, None]:
        for document in self.fetch(query, **kwargs):
            yield document

    @staticmethod
    def simple_pdf_parser(pdf_path: str) -> tuple[str, Dict[str, str]]:        
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pdf_pages = []
        for page in reader.pages:
            # extract text
            extracted_text = page.extract_text()

            # process text
            processed_lines = []
            for line in extracted_text.split("\n"):
                line = line.strip()
                if not line:
                    # line is empty
                    continue
                processed_lines.append(line)

            pdf_pages.append("\n".join(processed_lines))
        
        pdf_text = "\n".join(pdf_pages)

        pdf_meta = reader.metadata or {}
        return pdf_text, pdf_meta

