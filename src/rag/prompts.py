from typing import List, Dict
from rag.models import Document

RAG_PROMPT_STANDARD = '''You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
 \nQuestion: {query}
 \nContext: {context}
'''

RAG_SYSTEM_STANDARD = '''You are a helpful assistant that answers a query using the given query'''

class PromptFormatter:
    def __init__(self, system_prompt: str=None):
        """
        Initializes the PromptFormatter with an optional system prompt.
        Args:
            system_prompt: Optional initial system message for LLM context.
        """
        self.system_prompt = system_prompt
        self.messages = []  # Stores the formatted messages list

        # Initialize with the system prompt if provided
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_documents(self, docs: List[Document], method="concatenate", use_metadata=False, metadata_fields=None):
        """
        Adds documents to the messages using a specified combination method.
        Args:
            docs: List of Document objects.
            method: The method to combine documents ('concatenate', 'summary', 'bullet_points').
            use_metadata: Whether to include metadata in the messages.
            metadata_fields: List of metadata fields to include if use_metadata is True.
        """
        if not docs:
            raise ValueError("No documents provided to add.")
        
        if method == "concatenate":
            self._add_concatenated_docs(docs, use_metadata, metadata_fields)
        elif method == "summary":
            self._add_summarized_docs(docs, use_metadata, metadata_fields)
        elif method == "bullet_points":
            self._add_bulleted_docs(docs, use_metadata, metadata_fields)
        else:
            raise ValueError(f"Unknown combination method '{method}'.")

    def add_user_prompt(self, prompt: str):
        """
        Adds a user prompt to the messages list.
        Args:
            prompt: The user's input question or prompt.
        """
        if not prompt:
            raise ValueError("User prompt cannot be empty.")
        self.messages.append({"role": "user", "content": prompt})

    def get_messages(self):
        """
        Retrieves the current list of formatted messages.
        Returns:
            A list of message dictionaries formatted for the LLM.
        """
        return self.messages

    def reset(self, new_system_prompt: str=None):
        """
        Clears the message history, allowing a fresh start for a new conversation.
        Args:
            new_system_prompt: Optional new system message to start the conversation.
        """
        self.messages = []
        if new_system_prompt:
            self.system_prompt = new_system_prompt
            self.messages.append({"role": "system", "content": self.system_prompt})

    ### Private methods for different combination strategies ###

    def _add_concatenated_docs(self, docs: List[Document], use_metadata, metadata_fields):
        """
        Concatenates document contents and adds them as a single assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each document's content.
            metadata_fields: Specific metadata fields to include (e.g., ['title']).
        """
        combined_content = ""
        for doc in docs:
            content = doc.text
            if use_metadata and hasattr(doc, 'metadata'):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"{metadata}\n{content}"
            combined_content += content + "\n\n"

        self.messages.append({"role": "assistant", "content": combined_content.strip()})

    def _add_summarized_docs(self, docs: List[Document], use_metadata, metadata_fields):
        """
        Adds a summary of each document as an individual assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each summary.
            metadata_fields: Specific metadata fields to include in the summary.
        """
        for doc in docs:
            content = self._summarize(doc.text)  # Replace with actual summarization if available
            if use_metadata and hasattr(doc, 'metadata'):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"{metadata}\n{content}"
            self.messages.append({"role": "assistant", "content": content})

    def _add_bulleted_docs(self, docs: List[Document], use_metadata, metadata_fields):
        """
        Formats each document content as a bulleted list in a single assistant message.
        Args:
            docs: List of Document objects.
            use_metadata: If True, include specified metadata in each bullet point.
            metadata_fields: Specific metadata fields to include in each bullet point.
        """
        bulleted_content = ""
        for doc in docs:
            content = f"- {doc.text}"
            if use_metadata and hasattr(doc, 'metadata'):
                metadata = self._extract_metadata(doc, metadata_fields)
                content = f"- {metadata}: {doc.text}"
            bulleted_content += content + "\n"

        self.messages.append({"role": "assistant", "content": bulleted_content.strip()})

    def _extract_metadata(self, doc, fields):
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

    def _summarize(self, content):
        """
        Placeholder summarization method. Replace with a more sophisticated summarization if needed.
        Args:
            content: Text content of the document.
        Returns:
            A truncated or summarized version of the content.
        """
        # Simple truncation for example purposes
        return content[:100] + "..." if len(content) > 100 else content
