from ast import In
from random import random
from rag.operate import (
    ask_simple_llm,
    ask_simple_rag,
    build_vectorstore_index_from_mongo_db,
)
from rag.llm import OpenAIEmbeddingModel, OpenAILLM
from rag.vectorstore.in_memory import InMemoryVectorStore
from rag.vectorstore.chroma import ChromaVectorStore
from rag.index.vectorstore_index import VectorStoreIndex
from rag.text_splitters import ContextualTextSplitter

import gradio as gr
import asyncio
from datetime import datetime
from pathlib import Path


inmemory_vectorstore_dir = Path("tmp/inmemory")



embedding_model = OpenAIEmbeddingModel()
llm = OpenAILLM()

vectorstore1 = None
index1 = None

def index_1_information():
    if index1 is not None:
        return f"Index has {index1.vectorstore.size()} documents"
    else:
        return "Index not built yet"

# vectorstore2 = None
# index2 = None



async def handle_select_dataset(dataset: str):
    if dataset == "Deepseek News":
        return (
            "30 news article on Deepseek, scraped on FT, NYT, and NewsAPI on 01/27/2025"
        )
    elif dataset == "Financial Times":
        return "Search on Financial Times"

async def handle_select_vectorstore(path: str):
    global vectorstore1, index1

    vectorstore1 = InMemoryVectorStore.load_pickle(path)
    index1 = VectorStoreIndex(vectorstore=vectorstore1)

    gr.Info("Index built successfully", duration=5)

    return True # generating a random number to rerender the index_1_state

async def handle_build_index(dataset: str) -> bool:
    """build index based on the selected dataset. Currently only supports Deepseek News

    Args:
        dataset (str): type of data to build index for

    Raises:
        gr.Error: if dataset is not supported

    Returns:
        bool: True if index is built successfully
    """
    global embedding_model, llm
    global vectorstore1, index1

    if dataset == "Deepseek News":
        vectorstore1 = InMemoryVectorStore(
            embedding_model=embedding_model,
        )
        gr.Info("Building index for Deepseek News...", duration=5)

        # build index
        index1 = await build_vectorstore_index_from_mongo_db(
            embedding_model, vectorstore1, collection_name="news_search_2025-01-27"
        )

        # save vectorstore
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        vectorstore1.save_pickle(inmemory_vectorstore_dir / f"deepseek_news_{timestamp}.pickle")
        gr.Info("Index built successfully", duration=5)

        return True # generating a random number to rerender the index_1_state
    else:
        raise gr.Error(f"Dataset {dataset} not supported yet")



async def handle_submit(query):
    global embedding_model, llm
    global index1

    if index1 is None:
        raise gr.Error("Index not built yet")

    llm_response, simple_rag_response = await asyncio.gather(
        ask_simple_llm(query, llm),
        ask_simple_rag(query, index1, embedding_model, llm),
    )

    # TODO: add context metadata (e.g. headlines)
    simple_rag_contexts = [doc.text for doc in simple_rag_response["contexts"]]
    simple_rag_text = simple_rag_response["response"]

    return (llm_response, simple_rag_text, "\n\n".join(simple_rag_contexts))


with gr.Blocks() as demo:
    # session states
    index_1_state = gr.State(index1 is not None) # kinda a cheat state to trigger re-rendering
    

    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("## RAG Demo Comparator")

        state_display = gr.Textbox(label="Index 1 is set?", value=index_1_state.value, interactive=False)
        # with gr.Column(scale=0):
        # @gr.render(inputs=index_1_state)
        # def show_index_1_is_built(state):
        #     global index1
        #     if index1 is not None:
        #         gr.Markdown(value=lambda: f"{index1.vectorstore.size()} documents")
        #         gr.Image(
        #             value="gradio_app/assets/database-img.png",
        #             width=50,
        #             height=50,
        #             interactive=False,
        #             show_download_button=False,
        #             show_fullscreen_button=False,
        #             container=False,
        #             label="Index 1",
        #         )
        #     else:
        #         gr.Markdown("Index not built yet")
        
        # update the state_display when index_1_state changes
        index_1_state.change(fn=lambda x: x, inputs=[index_1_state], outputs=[state_display])
   

    with gr.Tab("Build Index"):
        with gr.Row():
            with gr.Column():
                # ===== DATASET SELECTION =====
                gr.Markdown("### Choose your dataset")
                dataset_select = gr.Dropdown(
                    label="Dataset",
                    choices=["Deepseek News", "Financial Times"],
                    value=None,
                    interactive=True,
                    allow_custom_value=False,
                )
                build_index_btn = gr.Button("Build Index")

                # dataset description/actions
                dataset_description = gr.Textbox(
                    label="Dataset Description", interactive=False
                )

                # when user selects a dataset, show the description of that dataset
                dataset_select.change(
                    handle_select_dataset,
                    inputs=[dataset_select],
                    outputs=[dataset_description],
                )

                # when user clicks build index, build the index.
                build_index_btn.click(
                    fn=handle_build_index,
                    inputs=[dataset_select],
                    outputs=[index_1_state],
                )

            with gr.Column():
                # ===== VECTORSTORE SELECTION =====
                gr.Markdown("### Load past vectorstore")
                vectorstore_select = gr.FileExplorer(
                    glob="*.pickle",
                    file_count="single",
                    root_dir=inmemory_vectorstore_dir,
                    label="Select vectorstore",
                    interactive=True
                )
                select_vs_btn = gr.Button("Select Vectorstore")

                # when user selects a vectorstore, load it
                select_vs_btn.click(
                    handle_select_vectorstore,
                    inputs=[vectorstore_select],
                    outputs=[index_1_state],
                )

    with gr.Tab("RAG Search"):
        query_input = gr.Textbox(label="Ask your question:", submit_btn="Submit")
        # submit_button = gr.Button("Submit")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Simple LLM Response")
                response_1 = gr.Textbox(label="LLM Response", interactive=False)
            with gr.Column():
                gr.Markdown("### Standard RAG Response")
                response_2 = gr.Textbox(label="RAG Response", interactive=False)
                show_cxt_2_btn = gr.Button("Click to see context")
                context_2 = gr.Textbox(visible=False, label="Contexts", lines=20)

                show_cxt_2_btn.click(
                    fn=lambda: gr.update(visible=False), outputs=show_cxt_2_btn
                )
                show_cxt_2_btn.click(
                    fn=lambda: gr.update(visible=True), outputs=context_2
                )
            with gr.Column():
                gr.Markdown("### Contextual RAG Response")
                response_3 = gr.Textbox(label="RAG Response", interactive=False)
                show_cxt_3_btn = gr.Button("Click to see context")
                context_3 = gr.Textbox(visible=False, label="Contexts", lines=20)

                show_cxt_3_btn.click(
                    fn=lambda: gr.update(visible=False), outputs=show_cxt_3_btn
                )
                show_cxt_3_btn.click(
                    fn=lambda: gr.update(visible=True), outputs=context_3
                )

            # with gr.Column():

            query_input.submit(
                handle_submit,
                inputs=[query_input],
                outputs=[response_1, response_2, context_2,]# response_3, context_3],
            )

demo.launch()
