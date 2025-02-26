from typing import List
import re
from abc import ABC, abstractmethod
from kruppe.models import Document, Query

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


 # ------------ DOCUMENT/CHUNK INFORMATION EXTRACTION ------------
PROMPTS["extract_document_entities"] = """Given a text document, identify all entities of this document.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity.
- entity_type: One of the following types: [Firm, Industry, Country]
- entity_description: Short, succinct description of the entity's background and attributes\
 using only information from the provided text. Focus on areas related to the entity's business,\
 for example how it makes money, its business strength or weakness, its management style, its history, etc.
Format each entity as ("entity"|<entity_name>|<entity_type>|<entity_description>)

2. Return output in English as a single list of all entities identified in steps 1.

-Example-
Input:
<document>
...
</document>

Output:
("entity"|General Motors|Firm|General Motors is a major American automobile manufacturer known\
 for producing vehicles and automotive parts. It has been significantly impacted by US tariffs\
 on trading partners, causing its stock to fluctuate negatively when tariff news emerges and then\
 recover when those tariffs are postponed.)
("entity"|Ford|Firm|Ford is a prominent American automotive company tasked with manufacturing\
 vehicles and automotive components. The firm has experienced stock impacts due to the uncertainty\
 surrounding US tariffs on imports, similar to General Motors, suffering declines but showing\
 resilience with tariff postponement.)

-Real Data-
Input:
<document>
{WHOLE_DOCUMENT}
</document>

Output:
"""

PROMPTS["extract_chunk_events"] = """-Goal-
Given a chunk of a text document and a list of entities, identify\
 all events in the text chunk and determine the relationships between the events and the entities.

-Steps-
1. Identify all events. For each identified event, extract the following information:
- event_name: Name of the event.
- event_type: One of the following types: [Business Risk, Business Opportunity, Regulatory Risk, Regulatory Opportunity, Geopolitical Risk, Geopolitical Opportunity]
- event_description: Comprehensive description of the event's attributes and activities
Format each entity as ("event"|<event_name>|<event_type>|<event_description>)

2. For event identified in step 1 and entities data, identify all pairs of (event, entity).\
 For each pair of related entities and events, extract the following information:
- event_name: name of the event
- entity_name: name of the entity that is involved in the event. The entity could have caused the event, or be affected by the event.
- relationship_description: explanation to why you think the event and the entity are related.
- relationship_strength: a numeric score indicating strength of the relationship between the entity and the event. 1 indicates a weak relationship, 10 indicates a strong relationship.
Format each pair as ("relationship"|<event_name>|<entity_name>|<relationship_description>|<relationship_strength>)

4. Return output in English as a single list of all events and relationships identified in steps 1 and 2.

-Example-
Input:
<entities>
("entity"|Morgan Stanley Investment Management|Firm|A global financial services firm that provides\
 investment management and a range of advisory services, including asset management and financial advisory,\
 with a focus on enhancing the investment experience of its clients."),
 ("entity"|Diageo|Firm|A multinational beverage company known for its premium alcoholic drinks,\
 including spirits and beers, and recognized for its marketing strategies and brand portfolio."),
</entities>
<chunk>
...
</chunk>

Output:
("event"|HEDGE FUND SHORTING|Business Risk|The practice of hedge funds increasingly betting against certain stocks, particularly those exposed to tariff risks, reflecting investor sentiment towards potential losses in those sectors."),
("relationship"|HEDGE FUND SHORTING|Goldman Sachs|Goldman Sachs's research highlights the trend among hedge funds to short stocks exposed to tariff risks, indicating a direct observation and response to market conditions by the firm.|7),
("relationship"|HEDGE FUND SHORTING|Diageo|Diageo's stocks have seen a drop as a result of greater short selling by hedge funds, reflecting concerns over the beverage industry’s exposure to economic disruptions from tariffs.|7),

-Real Data-
Input:
<entities>
{ENTITIES}
</entities>
<chunk>
{CHUNK_CONTENT}
</chunk>

Output:
"""

PROMPTS["extract_questions"] = """-Goal-
Find all questions that this document can answer.\
 These questions will help retrieve the right information in a retrieval-augmented generation (RAG) system.

For each question, extract:
- question: A question that can be answered using only information from this document. Keep it general.
    - Good example: “How is Netflix affected by rising interest rates?”
    - Bad example: “What does the author Thomas Shelby think about interest rates and Netflix?”
- original_text: The exact part of the document that answers the question.
- category: The type of question. Acceptable categories are: [Current Event, Historical Event, Concepts, Analysis].\
 Please see the definition for each category below.

Definition of acceptable question categories:
- Current Event: Asking for facts (NOT analysis) about a recent event.
- Historical Event: Asking for facts (NOT analysis) about a past event.
- Concepts: Asking for general ideas, not specific facts or opinions. Example: “How does inflation affect tech companies?”
- Analysis: Asking for analysis or opinions on specific facts. Example: “Why did Congress pass the 99th Amendment forcing Americans to become cyborgs?”

Format output as follows:

QUESTION 1
question: [Insert question]
original_text: [Insert original text]
category: [Insert category]

QUESTION 2
question: [Insert question]
original_text: [Insert original text]
category: [Insert category]"""

# ------------ GENERATE SUB-QUESTIONS FROM QUERY ------------
PROMPTS["query_generate_questions"] = """-Goal-
Given a financial query, generate sub questions that, when answered, will help generate a more comprehensive answer.

-Steps-
1. Identify all entities that are referenced in the question. Extract the following factors:
- entity_name: Name of the entity
- entity_type: One of the following types: [Firm, Industry, Country, Individual]
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. For each entity classified as "firm", generate questions that will help build a better picture to understand the firm. For each question, produce the following:
- question_statement: the question itself
- question_type: the type of question. Valid question types are: ["Firm History", "Strategic Posture", "Market Environment", "Opportunities", "Threats", "Strengths", "Weaknesses", "Conceptual", "Street's View"]
Format each question as ("question"|<question_statement>|<question_type>)

3. Return output in English as a single list of all entities and questions identified in steps 1 and 2.

-Examples-
Query: How will firing nuclear weapons into Neptune affect diplomatic relationships between the United States Government and the Knights of the Sun?

Output:
(entity|United States Government|Country)
(entity|Knights of the Sun|Firm)
(question|How does the Knights of the Sun make money?|Firm History)

-Real Input-
Query: How will the repeal of the US anti-corruption law affect giant U.S. banks' (like JPMorgan) business practice?

Output:
"""

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
