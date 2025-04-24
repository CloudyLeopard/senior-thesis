import asyncio
from kruppe.llm import OpenAILLM, OpenAIEmbeddingModel
from kruppe.functional.rag.vectorstore.chroma import ChromaVectorStore
from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.functional.docstore.mongo_store import MongoDBStore
from kruppe.functional.rag.retriever.simple_retriever import SimpleRetriever
from kruppe.functional.rag.retriever.fusion_retriever import QueryFusionRetriever
from kruppe.functional.ragquery import RagQuery
from kruppe.functional.llmquery import LLMQuery
from kruppe.functional.newshub import NewsHub
from kruppe.functional.finhub import FinHub
from kruppe.data_source.news.nyt import NewYorkTimesData
from kruppe.data_source.news.ft import FinancialTimesData
from kruppe.data_source.news.newsapi import NewsAPIData
from kruppe.data_source.finance.yfin import YFinanceData
from kruppe.llm import OpenAILLM

# Initialize LLM and embedding model
llm = OpenAILLM(model="gpt-4.1-mini")
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")

# Initialize vector store
vectorstore = ChromaVectorStore(
    embedding_model=embedding_model,
    collection_name="kruppe_librarian",
    persist_path="/Volumes/Lexar/Daniel Liu/vectorstores/kruppe_librarian"
)
index = VectorStoreIndex(llm=llm, vectorstore=vectorstore)

# Initialize document store
unique_indices = [['title', 'datasource']]
docstore = None  # Will be initialized asynchronously

async def initialize_storage():
    global docstore
    docstore = await MongoDBStore.acreate_db(
        db_name="kruppe_librarian",
        collection_name="kruppe_librarian",
        unique_indices=unique_indices
    )

# Initialize storage immediately
asyncio.run(initialize_storage()) 


# Initialize query engines
retriever = SimpleRetriever(index=index, top_k=10)
#  or use query fusion retriever

rag_query_engine = RagQuery(llm=llm, retriever=retriever)
llm_query_engine = LLMQuery(llm=llm)

# Initialize news sources
news_sources = [
    NewYorkTimesData(headers_path="/Users/danielliu/Workspace/fin-rag/.nyt-headers.json"),
    FinancialTimesData(headers_path="/Users/danielliu/Workspace/fin-rag/.ft-headers.json"),
    NewsAPIData()
]
news_hub = NewsHub(news_sources=news_sources[:])

# Initialize finance source
fin_hub = FinHub(
    fin_source=YFinanceData(),
    llm=llm
)

# Define all available tools
ALL_TOOLS = {
    "RAG Query": rag_query_engine.rag_query,
    "LLM Query": llm_query_engine.llm_query,
    "News Search": news_hub.news_search,
    "Recent News": news_hub.news_recent,
    "News Archive": news_hub.news_archive,
    "Company Background": fin_hub.get_company_background,
    "Company Income Statement": fin_hub.get_company_income_stmt,
    "Company Balance Sheet": fin_hub.get_company_balance_sheet,
    "Financial Analysis": fin_hub.analyze_company_financial_stmts
} 