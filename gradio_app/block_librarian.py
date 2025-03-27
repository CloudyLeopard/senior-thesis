import gradio as gr
import chromadb

# ----- KRUPPE IMPORTS ----- #
# import essentials
from kruppe.llm import OpenAILLM, OpenAIEmbeddingModel

# data sources to use
from kruppe.data_source.news.nyt import NewYorkTimesData

# import index and rag related stuff
from kruppe.functional.rag.vectorstore.chroma import ChromaVectorStore, get_chroma_collection_names
from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.functional.docstore.mongo_store import MongoDBStore

# import librarian
from kruppe.algorithm.librarian import Librarian

# initializing news sources
news_sources = [
    NewYorkTimesData(headers_path="/Users/danielliu/Workspace/fin-rag/.nyt-headers.json"),

]
news_sources_map = {source.source: source for source in news_sources}

# initializing vectorstore client
vectorstore_path = '/Volumes/Lexar/Daniel Liu/vectorstores/kruppe_librarian'
chroma_client = chromadb.PersistentClient(path=vectorstore_path)

collection_names = get_chroma_collection_names(chroma_client)

# Global states
librarian = None # Librarian object

def get_librarian():
    """For other blocks to access the global librarian object"""
    global librarian
    return librarian

async def initialize_librarian(param_model, param_embed_model, param_coll_name, param_news_source, param_num_retries, param_relevance_score_threshold, param_resource_rank_threshold, param_num_rsc_per_retrieve):
    global librarian

    # initialize the LLM model
    llm = OpenAILLM(model=param_model)
    embed_model = OpenAIEmbeddingModel(model=param_embed_model)

    # initialize the vectorstore index
    vectorstore = ChromaVectorStore(
        embedding_model=embed_model,
        collection_name=param_coll_name,
        client=chroma_client
    )
    index = VectorStoreIndex(llm=llm, vectorstore=vectorstore)

    # initialize/create doc source
    unique_indices = [['title', 'datasource']] # NOTE: necessary to avoid duplicates
    docstore = await MongoDBStore.acreate_db(
        db_name='kruppe_librarian',
        collection_name=param_coll_name,
        unique_indices=unique_indices
    )

    # initialize the data source
    news_source = news_sources_map[param_news_source]

    # initialize the librarian
    librarian = Librarian(
        llm=llm,
        docstore=docstore,
        index=index,
        news_source=news_source,
        num_retries=param_num_retries,
        relevance_score_threshold=param_relevance_score_threshold,
        resource_rank_threshold=param_resource_rank_threshold,
        num_rsc_per_retrieve=param_num_rsc_per_retrieve
    )

    return f"Librarian Initialized! Currently has {librarian.document_count} documents"

async def execute_librarian(info_request, arg_top_k, arg_llm_restrict_time, arg_start_time, arg_end_time):
    global librarian

    kwargs = {
        "top_k": arg_top_k,
        "llm_restrict_time": arg_llm_restrict_time,
        "start_time": arg_start_time,
        "end_time": arg_end_time
    }

    ret_docs = await librarian.execute(information_desc=info_request, **kwargs)

    return "\n\n".join([doc.text for doc in ret_docs])

def create_librarian_block():
    with gr.Blocks() as block:
        gr.Markdown("# Librarian")
        gr.Markdown("## Initialize Librarian")
        with gr.Group():
            gr.Markdown('### `Librarian` Configuration')

            with gr.Row():
                param_model = gr.Dropdown(
                    label="Select LLM Model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    multiselect=False,
                    value="gpt-4o-mini",
                    interactive=True,
                    allow_custom_value=False,
                )

                param_embed_model = gr.Dropdown(
                    label="Select Embedding Model",
                    choices=["text-embedding-3-small", "text-embedding-3-large"],
                    multiselect=False,
                    value="text-embedding-3-small",
                    interactive=True,
                    allow_custom_value=False,
                )

                param_coll_name = gr.Dropdown(
                    label="Select the Librarian's index",
                    info="Choosing a custom value will create a new index",
                    choices=collection_names,
                    multiselect=False,
                    interactive=True,
                    allow_custom_value=True,
                )

                param_news_source = gr.Dropdown(
                    label="Select News Source to scrape from",
                    choices=news_sources_map.keys(),
                    multiselect=False,
                    interactive=True,
                    allow_custom_value=False,
                )

            gr.Markdown('#### Lesser Configurations')
            with gr.Row():
                param_num_retries = gr.Number(2, label="Number of Retries", interactive=True)
                param_relevance_score_threshold = gr.Number(2, label="Relevance Score Threshold", minimum=1, maximum=3, step=1, interactive=True)
            with gr.Row():
                param_resource_rank_threshold = gr.Number(2, label="Resource Rank Threshold", minimum=1, maximum=3, step=1, interactive=True)
                param_num_rsc_per_retrieve = gr.Number(2, label="Number of Resources to Retrieve", interactive=True)

        init_button = gr.Button("Initialize Librarian", variant='primary')
        output_config = gr.Textbox(label="Initialization Status", interactive=False)
        
        init_button.click(initialize_librarian, inputs=[param_model, param_embed_model, param_coll_name, param_news_source, param_num_retries, param_relevance_score_threshold, param_resource_rank_threshold, param_num_rsc_per_retrieve], outputs=[output_config])
        
        gr.Markdown("## Execute Librarian")

        info_request = gr.Textbox(label="Info Request", placeholder="Enter a piece of information you want from the Librarian")
        
        with gr.Group():
            gr.Markdown("### Librarian Execution Configuration")
            arg_top_k = gr.Number(label="Top K", value=10, interactive=True)
            arg_llm_restrict_time = gr.Checkbox(label="Use LLM to Restrict Publication Time", interactive=True)
            arg_start_time = gr.DateTime(label="Start Time", interactive=True)
            arg_end_time = gr.DateTime(label="End Time", interactive=True)
        
        execute_button = gr.Button("Execute Librarian", variant='huggingface')

        ret_contexts = gr.Textbox(label="Info Output", lines=10, interactive=False)

        execute_button.click(execute_librarian, inputs=[info_request, arg_top_k, arg_llm_restrict_time, arg_start_time, arg_end_time], outputs=[ret_contexts])

    return block