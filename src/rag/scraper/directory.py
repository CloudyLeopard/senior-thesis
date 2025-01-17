from typing import List, Dict
import pathlib
import logging
from pydantic import Optional, field_validator, model_validator

from rag.scraper.base_source import BaseDataSource
from rag.models import Document

logger = logging.getLogger(__name__)

class DirectoryData(BaseDataSource):
    path: Optional[pathlib.Path] = None
    input_files: Optional[List[pathlib.Path]] = None

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
    @classmethod
    def validate_either_path_or_input_files(cls, values):
        if not (values["path"] is None or values["input_files"] is None):
            raise ValueError("Must provide either path or input_files")
        return values

    def __init__(self, **data):
        super().__init__(**data)
        self.source = "Local Directory"
        if self.path is not None:
            self._file_generator = self.path.rglob("*")
        else:
            self._file_generator = self.input_files

    def fetch(self, query: str = None, **kwargs) -> List[Document]:
        """given path to data folder, fetch text files in 
        subdirectory with name matching query"""
        
        documents = []
        for file_path in self._file_generator:
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
                documents.append(Document(text=txt, metadata=metadata))
            if file_path.suffix == ".pdf":
                pdf_text, pdf_meta = self.simple_pdf_parser(file_path.as_posix())
                metadata = self.parse_metadata(
                    query="NA",
                    name=file_path.name,
                    path=file_path.as_posix(),
                    publication_time = pdf_meta.get("creation_date"),
                    title = pdf_meta.get("title")
                )
                documents.append(Document(text=pdf_text, metadata=metadata))

        return documents

    async def async_fetch(self, query: str, num_results: int = None, **kwargs) -> List[Document]:
        """N/A. Calls on sync. fetch function"""
        return self.fetch(query, num_results)

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

