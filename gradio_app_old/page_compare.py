# from kruppe.operate import (
#     ask_simple_llm,
#     ask_simple_rag,
#     build_index_from_mongo_db,
# )
from kruppe.llm import OpenAIEmbeddingModel, OpenAILLM, NYUOpenAIEmbeddingModel, NYUOpenAILLM
from kruppe.functional.rag.vectorstore.in_memory import InMemoryVectorStore
from kruppe.functional.rag.vectorstore.chroma import ChromaVectorStore
from kruppe.functional.rag.index.base_index import BaseIndex
from kruppe.functional.rag.index.vectorstore_index import VectorStoreIndex
from kruppe.functional.rag.index.contextual_index import ContextualVectorStoreIndex

import gradio as gr
from datetime import datetime
from pathlib import Path

from kruppe.models import Response

inmemory_vectorstore_dir = Path("tmp/inmemory")
(inmemory_vectorstore_dir / "Vectorstore Index").mkdir(parents=True, exist_ok=True)
(inmemory_vectorstore_dir / "Contextual Index").mkdir(parents=True, exist_ok=True)

embedding_model = OpenAIEmbeddingModel()
llm = OpenAILLM()

vectorstore1 = None
index1 = None

vectorstore2 = None
index2 = None


# =========================
# ------- HANDLERS --------
# =========================

# ---- MODEL HANDLERS -----

async def handle_change_model(model: str, api_key: str):
    global llm, embedding_model
    if model == "OpenAI":
        llm = OpenAILLM(api_key=api_key) if api_key else OpenAILLM()
        embedding_model = OpenAIEmbeddingModel(api_key=api_key)
        gr.Info("Model changed to OpenAI", duration=5)
    elif model == "NYU OpenAI":
        llm = NYUOpenAILLM(api_key=api_key) if api_key else NYUOpenAILLM()
        embedding_model = NYUOpenAIEmbeddingModel(api_key=api_key)
        gr.Info("Model changed to NYU OpenAI", duration=5)

# ---- DATASET HANDLERS ---

async def handle_select_dataset(dataset: str):
    if dataset == "Deepseek News":
        return (
            "30 news article on Deepseek, scraped on FT, NYT, and NewsAPI on 01/27/2025"
        )
    elif dataset == "Financial Times":
        return "Search on Financial Times"

# ---- INDEX HANDLERS ----

async def build_index_helper(dataset: str, index: BaseIndex):
    if dataset is None:
        raise gr.Error("Please select a dataset first")
    elif dataset == "Deepseek News":
        gr.Info("Building index for Deepseek News...", duration=5)

        # build index
        # await build_index_from_mongo_db(index=index, collection_name="news_search_2025-01-27")

        # save vectorstore
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        vectorstore1.save_pickle(inmemory_vectorstore_dir / f"deepseek_news_{timestamp}.pickle")
        gr.Info("Index built successfully", duration=5)

        return True # generating a random number to rerender the index_1_state
    else:
        raise gr.Error(f"Dataset {dataset} not supported yet")

async def handle_build_index_1(dataset: str) -> str:
    global embedding_model, llm
    global vectorstore1, index1

    vectorstore1 = InMemoryVectorStore(embedding_model=embedding_model)
    index1 = VectorStoreIndex(vectorstore=vectorstore1)

    await build_index_helper(dataset, index1)

     # save vectorstore
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # create directory if not exists
    path = inmemory_vectorstore_dir / "Vectorstore Index" / f"{dataset}_{timestamp}.pickle"
    vectorstore1.save_pickle(path)

    return path.stem

async def handle_build_index_2(dataset: str) -> str:
    global embedding_model, llm
    global vectorstore2, index2

    vectorstore2 = InMemoryVectorStore(embedding_model=embedding_model)
    index2 = ContextualVectorStoreIndex(vectorstore=vectorstore2, llm=llm)

    await build_index_helper(dataset, index2)

     # save vectorstore
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # create directory if not exists
    path = inmemory_vectorstore_dir / "Contextual Index" / f"{dataset}_{timestamp}.pickle"
    vectorstore2.save_pickle(path)

    return path.stem

async def handle_select_vectorstore_1(path: str):
    global vectorstore1, index1

    vectorstore1 = InMemoryVectorStore.load_pickle(path=path)
    index1 = VectorStoreIndex(vectorstore=vectorstore1)

    gr.Info("Index built successfully", duration=5)

    return Path(path).stem 

async def handle_select_vectorstore_2(path: str):
    global vectorstore2, index2
    global llm

    vectorstore2 = InMemoryVectorStore.load_pickle(path=path)
    index2 = ContextualVectorStoreIndex(vectorstore=vectorstore2, llm=llm)

    gr.Info("Index built successfully", duration=5)

    return Path(path).stem  

# ---- QUERY HANDLERS ----

async def handle_submit_1(query):
    global llm

    if not query:
        raise gr.Error("Please enter a query first")
    
    # llm_response = await ask_simple_llm(query, llm)
    # return llm_response.text

    return "blah"

async def handle_submit_2(query):
    global embedding_model, llm
    global index1

    if not query:
        raise gr.Error("Please enter a query first")

    if index1 is None:
        raise gr.Error("Index not built yet")

    # simple_rag_response = await ask_simple_rag(query, index1, embedding_model, llm)


    # # TODO: add context metadata (e.g. headlines)
    # simple_rag_contexts = [doc.text for doc in simple_rag_response["contexts"]]
    # simple_rag_text = simple_rag_response["response"]

    # return (simple_rag_text, "\n\n".join(simple_rag_contexts))

    return ("blahblah", "contextshahaha")

async def handle_submit_3(query):
    global embedding_model, llm
    global index2

    if not query:
        raise gr.Error("Please enter a query first")

    if index2 is None:
        raise gr.Error("Index not built yet")

    # simple_rag_response = await ask_simple_rag(query, index2, embedding_model, llm)

    # # TODO: add context metadata (e.g. headlines)
    # simple_rag_contexts = [doc.text for doc in simple_rag_response["contexts"]]
    # simple_rag_text = simple_rag_response["response"]

    # return (simple_rag_text, "\n\n".join(simple_rag_contexts))

    return ("blahblah", "contextshahaha")


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("# RAG Demo Comparator")


    with gr.Tab("Select Dataset"):
        # ===== DATASET SELECTION =====
        gr.Markdown("## Step 1: Choose your dataset")
        dataset_select = gr.Dropdown(
            label="Dataset",
            choices=["Deepseek News", "Financial Times"],
            value=None,
            interactive=True,
            allow_custom_value=False,
        )

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

        # ===== MODEL SELECTION =====
        gr.Markdown("## Optional: Select your model")
        model_select = gr.Dropdown(
            label="Model",
            choices=["OpenAI", "NYU OpenAI"],
            multiselect=False,
            value="OpenAI",
            interactive=True,
            allow_custom_value=False,
        )
        model_api_key = gr.Textbox(label="API Key", placeholder="Use environment variable", interactive=True, type="password")
        change_model_btn = gr.Button("Change Model")

        change_model_btn.click(
            handle_change_model,
            inputs=[model_select, model_api_key],
        )
    with gr.Tab("LLM Search"):
        query_1 = gr.Textbox(label="Ask your question:", submit_btn="Submit")
        # submit_button = gr.Button("Submit")
        # with gr.Row():
        gr.Markdown("## Simple LLM Response")
        response_1 = gr.Textbox(label="LLM Response", interactive=False, lines=10)

        query_1.submit(
            handle_submit_1,
            inputs=[query_1], outputs=[response_1], show_progress=True
        )
    with gr.Tab("RAG Search"):
        gr.Markdown("## Step 2: Build Index")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Option 1: Build New Index from Dataset")
                build_index_1 = gr.Button("Build Index")
         
                # ===== VECTORSTORE SELECTION =====
                gr.Markdown("### Option 2: Load past vectorstore")
                vectorstore_select_1 = gr.FileExplorer(
                    glob="*.pickle",
                    file_count="single",
                    root_dir=inmemory_vectorstore_dir / "Vectorstore Index",
                    label="Select vectorstore",
                    interactive=True
                )
                select_vs_btn_1 = gr.Button("Select Vectorstore")

                
            with gr.Column():
                gr.Markdown("### Selected Index")
                selected_index_1_txt = gr.Textbox(value="None. Select an option from the left", interactive=False, lines=5, show_label=False, container=False)

            # when user clicks build index, build the index.
            build_index_1.click(
                fn=handle_build_index_1,
                inputs=[dataset_select],
                outputs=[selected_index_1_txt]
            )

            # when user selects a vectorstore, load it
            select_vs_btn_1.click(
                handle_select_vectorstore_1,
                inputs=[vectorstore_select_1],
                outputs=[selected_index_1_txt],
            )

        # with gr.Row():
        query_2 = gr.Textbox(label="Ask your question:", submit_btn="Submit")
        # submit_button = gr.Button("Submit")
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

        # with gr.Column():

        query_2.submit(
            handle_submit_2,
            inputs=[query_2],
            outputs=[response_2, context_2,]# response_3, context_3],
        )
    with gr.Tab("Contextual RAG"):
        gr.Markdown("## Step 2: Build Index")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Option 1: Build New Index from Dataset")
                build_index_2 = gr.Button("Build Index")

                
         
                # ===== VECTORSTORE SELECTION =====
                gr.Markdown("### Option 2: Load past vectorstore")
                vectorstore_select_2 = gr.FileExplorer(
                    glob="*.pickle",
                    file_count="single",
                    root_dir=inmemory_vectorstore_dir / "Contextual Index",
                    label="Select vectorstore",
                    interactive=True
                )
                select_vs_btn_2 = gr.Button("Select Vectorstore")

                
            with gr.Column():
                gr.Markdown("### Selected Index")
                selected_index_2_txt = gr.Textbox(value="None. Select an option from the left", interactive=False, lines=5, show_label=False, container=False)

            # when user clicks build index, build the index.
            build_index_2.click(
                fn=handle_build_index_2,
                inputs=[dataset_select],
                outputs=[selected_index_2_txt]
            )

            # when user selects a vectorstore, load it
            select_vs_btn_2.click(
                handle_select_vectorstore_2,
                inputs=[vectorstore_select_2],
                outputs=[selected_index_2_txt],
            )

        # with gr.Row():
        query_3 = gr.Textbox(label="Ask your question:", submit_btn="Submit")
        # submit_button = gr.Button("Submit")
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

        query_3.submit(
            handle_submit_3,
            inputs=[query_3],
            outputs=[response_3, context_3,]# response_3, context_3],
        )

demo.launch()
