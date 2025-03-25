from typing import List
import asyncio
from kruppe.embeddings import BaseEmbeddingModel, OpenAIEmbeddingModel
from kruppe.functional.rag.text_splitters import RecursiveTextSplitter
from kruppe.vector_storages import BaseVectorStorage, NumPyVectorStorage
from kruppe.functional.docstore.mongo_store import BaseDocumentStore, MongoDBStore
from kruppe.llm import BaseLLM, OpenAILLM
from kruppe.prompt_formatter import SimplePromptFormatter, RAGPromptFormatter, CustomPromptFormatter
from kruppe.tools.sources import DirectoryData
from kruppe.tools.searcher import NewsArticleSearcher

TOPIC_EXTRACTION_PROMPT = """Extract this query's main subject. \
Keep it three words max. As an example, \
if the query is 'What is Amazon's current stock performance?', \
the subject is 'Amazon'.

Query: {query}"""

WRITE_STORY_PROMPT_RAG = """Write a story about {topic}'s business model \
using the contexts below. Keep the generated story as a paragraph. Your story \
should adequately explain the justification for the business model and \
any related business decisions.

Your answer should be about {topic}'s business model as a whole. However, you can include additional information about {query}

Contexts:
{context}
"""

UPDATE_STORY_PROMPT_RAG = """Below are previously written stories about a {topic}'s business model, as well as additional events \
or information that has yet to be included in the story. Generate a new story that considers the new information, \
and update the story accordingly. You can either improve upon the existing story, or completely rewrite it. \
Keep the story as a paragraph. It should adequately explain the reasonining behind why certain business actions are done.

Your answer should be about {topic}'s business model as a whole. However, you can include additional information about {query}

Previous stories:
{story}

New contexts:
{context}

Updated story:
"""

FETCH_DOCUMENTS_PROMPT = """Given the story below, generate as much potential events that could be \
relevant to the story as possible. Topics could be business operation, macro events, industry changes, \
new technology, or anything creative. Topics are valid even if they are not directly related, but \
could be related indirectly or even insignificantly. Output each topic as a few words. Output a list \
of {k} topics separated by new line symbol and no leading bullet points.

Business stories:
{story}
"""

RELEVANCE_PROMPT = """News: {text} 
Does any part of the news connect to, even in a very remote way, the business story described below? 
Story: {story}

Options: Yes, No. Justify your answer. Keep your justification short, but logical.
Your answer is (Please always begin your response with Yes or No): """

INTERPRETER_PROMPT = """Given the story below, answer the following finance question: {query}

Story:
{story}

Answer: """

class StoryWriter:
    def __init__(
        self,
        query: str,
        searcher: NewsArticleSearcher,
        topic: str = None,
        llm: BaseLLM = None,
        vectorstore: BaseVectorStorage = None,
        documentstore: BaseDocumentStore = None,
    ):
        self.query = query
        self.searcher = searcher

        self.llm = llm or OpenAILLM()
        self.vectorstore = vectorstore or NumPyVectorStorage(OpenAIEmbeddingModel())
        self.documentstore = documentstore or MongoDBStore()
        self.text_splitter = RecursiveTextSplitter()
        self.topic = topic or self._extract_topic(query)

        self.story = ""
        self.stories = []

    def _extract_topic(self, query: str):
        prompt_formatter = SimplePromptFormatter()
        messages = prompt_formatter.format_messages(
            TOPIC_EXTRACTION_PROMPT.format(query=query)
        )
        return self.llm.generate(messages)

    async def initialize_story(self, files: List[str] = []):
        # === add documents to document store ===
        documents = []
        # process priority documents (user uploaded)
        priority_documents = DirectoryData(input_files=files).fetch()
        documents.extend(priority_documents)

        # TODO: implement additional document searches

        self.documentstore.save_documents(documents)

        # === index documents ===
        chunked_documents = self.text_splitter.split_documents(documents)
        await self.vectorstore.async_insert_documents(chunked_documents)

        # === initialize story (first version) ===
        contexts = await self.vectorstore.async_similarity_search(self.query, top_k=5)
        prompt_formatter = RAGPromptFormatter(prompt_template=WRITE_STORY_PROMPT_RAG)
        prompt_formatter.add_documents(contexts)
        messages = prompt_formatter.format_messages(self.query, topic=self.topic)
        self.story = await self.llm.async_generate(messages)
        self.stories.extend(self.story)
        return self.story

    async def generate_relevant_events(self) -> List[str]:
        # ====== generate new keywords =====
        system_prompt = "You generate relevant topics given a story about a business model."
        prompt_formatter = CustomPromptFormatter(
            prompt_template=FETCH_DOCUMENTS_PROMPT,
            system_prompt=system_prompt
        )
        messages = prompt_formatter.format_messages(
            k=15, story="\n\n".join(self.stories)
        )
        response = await self.llm.async_generate(messages)
        topics = [topic.strip() for topic in response.split("\n") if topic.strip()]
        return topics

    async def fetch_relevant_documents(self, topics: List[str]):
        # ====== fetch relevant documents =====
        documents = []
        for topic in topics:
            contexts = await self.searcher.async_search(topic, num_results=10)
            documents.extend(contexts)

        # ====== extract relevant events =====
        relevance_prompt_formatter = CustomPromptFormatter(
            system_prompt="You extract relevance from a story given a text.",
            prompt_template=RELEVANCE_PROMPT,
        )
        tasks = []
        for document in documents:
            messages = relevance_prompt_formatter.format_messages(
                text=document.text, story="\n\n".join(self.stories)
            )
            tasks.append(self.llm.async_generate(messages))

        relevance_results = await asyncio.gather(*tasks)
        relevant_documents = [
            doc for doc, relevance in zip(documents, relevance_results) 
            if relevance.startswith("Yes") or relevance.startswith("yes")
        ]

        self.documentstore.save_documents(relevant_documents)
        # TODO: THIS PART IS NOT RIGHT 
        # shouldn't be adding to the same document store (othrewise ill get repeated info)
        chunked_documents = self.text_splitter.split_documents(relevant_documents)
        await self.vectorstore.async_insert_documents(chunked_documents)

        return relevant_documents

    async def extend_story(self):
        contexts = await self.vectorstore.async_similarity_search(self.query, top_k=5)
        prompt_formatter = RAGPromptFormatter(prompt_template=UPDATE_STORY_PROMPT_RAG)
        prompt_formatter.add_documents(contexts)
        messages = prompt_formatter.format_messages(self.query, topic=self.topic, story="\n\n".join(self.stories))
        self.story = await self.llm.async_generate(messages)
        self.stories.extend(self.story)
        return self.story
    
    def as_interpreter(self):
        return StoryInterpreter(story=self.story)

class StoryInterpreter:
    def __init__(self, story, llm: BaseLLM = None):
        self.llm = llm or OpenAILLM()
        self.story = story
    
    async def ask(self, query: str):
        prompt_formatter = CustomPromptFormatter(prompt_template=INTERPRETER_PROMPT)
        messages = prompt_formatter.format_messages(query=query, story=self.story)
        response = await self.llm.async_generate(messages)
        return response
