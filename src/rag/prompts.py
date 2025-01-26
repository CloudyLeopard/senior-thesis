from typing import List
import re
from abc import ABC, abstractmethod
from rag.models import Document, Query

PROMPTS = {}

# ------------ DEFAULT RAG PROMPTS ------------

PROMPTS["system_standard"] = "You are a helpful assistant."

PROMPTS["rag_prompt_standard"] = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
 \nQuestion: {query}
 \nContext: {context}
"""

PROMPTS["rag_system_standard"] = "You are a helpful assistant that answers a query using the given contexts"

# ------------ CONTEXTUALIZE CHUNK PROMPTS ------------
# source: https://www.anthropic.com/news/contextual-retrieval
PROMPTS["contextualize_chunk_prompt"] = """<document> 
{WHOLE_DOCUMENT}
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{CHUNK_CONTENT}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# ============================================
# ------------ PROMPT FORMATTERS ------------
# ============================================

class BasePromptFormatter(ABC):
    @abstractmethod
    def __init__(self, system_prompt: str = None):
        pass

    @abstractmethod
    def format_messages(self, user_prompt: str):
        pass

    @abstractmethod
    def reset(self):
        pass

class CustomPromptFormatter(BasePromptFormatter):
    def __init__(self, prompt_template: str, system_prompt: str = PROMPTS["system_standard"]):
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    def format_messages(self, **kwargs):
        if any(key not in re.findall(r"{(.*?)}", self.prompt_template) for key in kwargs):
            raise ValueError(
                "Prompt template does not contain all the necessary fields for formatting."
            )
        
        user_prompt = self.prompt_template.format(**kwargs)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def reset(self, system_prompt: str = None, prompt_template: str = None):
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

class SimplePromptFormatter(BasePromptFormatter):
    def __init__(self, system_prompt: str = PROMPTS["system_standard"]):
        """
        Initializes the SimplePromptFormatter with an optional system prompt.

        Args:
            system_prompt: Optional initial system message for LLM context.
        """
        self.system_prompt = system_prompt

    def format_messages(self, user_prompt: str | Query):
        """
        Formats the user prompt into a list of messages accepted by the LLM.

        Args:
            user_prompt: The user's input question or prompt.

        Returns:
            A list of dictionaries where each dictionary contains a 'role' and a 'content' key.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def reset(self, new_system_prompt: str = None):
        """
        Resets the prompt formatter with an optional new system prompt.

        Args:
            new_system_prompt: The new system prompt.
        """
        self.system_prompt = new_system_prompt


class RAGPromptFormatter(BasePromptFormatter):
    def __init__(
        self,
        system_prompt: str = PROMPTS["rag_system_standard"],
        prompt_template="default",
        documents: List[Document] = [],
    ):
        # initialize with documents
        """
        Initializes the RAGPromptFormatter with optional system prompt, prompt template, and documents.

        Args:
            system_prompt: Optional initial system message for LLM context.
            prompt_template: Template for formatting the user prompt. Must contain 'query' and 'context' fields if 
                            not set to one of the acceptable choices: "default", .
            documents: List of Document objects to be used in the prompt formatting.
        
        Raises:
            ValueError: If a custom prompt template does not contain both 'query' and 'context' fields.
        """
        self.system_prompt = system_prompt
        self.documents = documents

        # initialize prompt template for formatting user prompt
        if prompt_template == "default":
            self.prompt_template = PROMPTS["rag_prompt_standard"]
        else:
            pattern = r"(?s)\{query\}.*\{context\}|\{context\}.*\{query\}"
            if not re.search(pattern, prompt_template):
                raise ValueError(
                    "Custom prompt template is not valid. Must have 'query' and 'context' fields"
                )
            self.prompt_template = prompt_template

    def add_documents(self, docs: List[Document]):
        """
        Adds documents to the messages.
        Args:
            docs: List of Document objects.
        """
        if not docs:
            raise ValueError("No documents provided to add.")

        self.documents.extend(docs)

    def format_messages(
        self,
        user_prompt: str | Query,
        method="concatenate",
        use_metadata=False,
        metadata_fields=None,
         **kwargs
    ):
        """
        Formats the user prompt and documents into a list of messages for LLM processing.

        Args:
            user_prompt: The user's input question or prompt.
            method: The method to combine documents, options are "concatenate", "summary", or "bullet_points".
            use_metadata: If True, includes specified metadata in the document content.
            metadata_fields: List of specific metadata fields to include.

        Returns:
            A list of dictionaries where each dictionary contains a 'role' and a 'content' key,
            representing the system and user messages formatted for the LLM.

        Raises:
            ValueError: If no documents are added to the prompt formatter or an unknown combination method is specified.
        """
        if not self.documents:
            raise ValueError("No documents added to the prompt formatter.")
        
        if isinstance(user_prompt, Query): 
            user_prompt = user_prompt.text

        if any(key not in re.findall(r"{(.*?)}", self.prompt_template) for key in kwargs):
            raise ValueError(
                "Prompt template does not contain all the necessary fields for formatting."
            )
        
        # Add system message
        messages = [{"role": "system", "content": self.system_prompt}]

        # Combine documents
        if method == "concatenate":
            context = self._combine_concatenate(
                self.documents, use_metadata, metadata_fields
            )
        elif method == "summary":
            context = self._combine_summary(
                self.documents, use_metadata, metadata_fields
            )
        elif method == "bullet_points":
            context = self._combine_bullet_points(
                self.documents, use_metadata, metadata_fields
            )
        else:
            raise ValueError(f"Unknown combination method '{method}'.")

        # Format RAG prompt
        prompt = self.prompt_template.format(query=user_prompt, context=context, **kwargs)

        # Add user message
        user_message = {"role": "user", "content": prompt}
        messages.append(user_message)

        return messages

    def reset(self, system_prompt: str = None):
        """
        Resets the prompt formatter with an optional new system prompt.
        
        Args:
            system_prompt: The new system prompt. If None, the default system prompt will be used.
        """
        self.documents = []

        if system_prompt:
            self.system_prompt = system_prompt


    ### Private methods for different combination strategies ###

    def _combine_concatenate(self, docs: List[Document], use_metadata, metadata_fields):
        """
        Concatenates document contents and adds them as a single assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each document's content.
            metadata_fields: Specific metadata fields to include (e.g., ['title']).
        Returns:
            Combined content of documents as a single string.
        """
        combined_content = ""
        for doc in docs:
            content = doc.text
            if use_metadata and hasattr(doc, "metadata"):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"{metadata}\n{content}"
            combined_content += content + "\n\n"
        return combined_content.strip()

    def _combine_summary(self, docs: List[Document], use_metadata, metadata_fields):
        """
        Adds a summary of each document as an individual assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each summary.
            metadata_fields: Specific metadata fields to include in the summary.
        Returns:
            Combined summaries of documents as a single string.
        """
        summary_content = ""
        for doc in docs:
            content = self._summarize(
                doc.text
            )  # Replace with actual summarization if available
            if use_metadata and hasattr(doc, "metadata"):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"{metadata}\n{content}"
            summary_content += content + "\n\n"
        return summary_content.strip()

    def _combine_bullet_points(
        self, docs: List[Document], use_metadata, metadata_fields
    ):
        """
        Formats each document content as a bulleted list in a single assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each bullet point.
            metadata_fields: Specific metadata fields to include in each bullet point.
        Returns:
            Bulleted list of documents as a single string.
        """
        bulleted_content = ""
        for doc in docs:
            content = f"- {doc.text}"
            if use_metadata and hasattr(doc, "metadata"):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"- {metadata}: {doc.text}"
            bulleted_content += content + "\n"
        return bulleted_content.strip()

    def _extract_metadata(self, doc: Document, fields: List[str]):
        """
        Helper method to extract specified metadata fields from a document.
        Args:
            doc: Document object with metadata.
            fields: List of fields to include in the metadata string.
        Returns:
            A string with selected metadata fields.
        """
        if not fields:
            return ""
        return ", ".join(f"{field}: {doc.metadata.get(field, '')}" for field in fields)

    def _summarize(self, content: str):
        """
        Placeholder summarization method. Replace with a more sophisticated summarization if needed.
        Args:
            content: Text content of the document.
        Returns:
            A truncated or summarized version of the content.
        """
        # Simple truncation for example purposes
        return content[:100] + "..." if len(content) > 100 else content
