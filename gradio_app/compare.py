from rag.operate import ask_simple_llm, ask_simple_rag, build_vectorstore_index_from_mongo_db
from rag.llm import OpenAIEmbeddingModel, OpenAILLM
from rag.vectorstore.in_memory import InMemoryVectorStore
from rag.vectorstore.chroma import ChromaVectorStore
from rag.index.vectorstore_index import VectorStoreIndex

import gradio as gr
import asyncio


embedding_model=OpenAIEmbeddingModel()
llm = OpenAILLM()
# vectorstore = InMemoryVectorStore.load_pickle("/Users/danielliu/Workspace/fin-rag/src/rag/tmp/vectorstore.pickle")
vectorstore = ChromaVectorStore(embedding_model=embedding_model, collection_name="financial-times", persist_path="/tmp/ft_chroma_vectorstore")
index = VectorStoreIndex(embedder=embedding_model, vectorstore=vectorstore)

# asyncio.run(build_vectorstore_index(embedding_model=embedding_model))

# async def ask_simple_llm(query, llm):
#     return "simple"

# async def ask_simple_rag(query, index, embedding_model, llm):
#     return {"response": "simple", "contexts": ["blah","heh", "meh"]}

async def handle_submit(query):
    llm_response, simple_rag_response = await asyncio.gather(
        ask_simple_llm(query, llm),
        ask_simple_rag(query, index, embedding_model, llm)
    )

    # TODO: add context metadata (e.g. headlines)
    simple_rag_contexts = [doc.text for doc in simple_rag_response["contexts"]]
    simple_rag_text = simple_rag_response["response"]

    return (llm_response,
            simple_rag_text, "\n\n".join(simple_rag_contexts)
            )

with gr.Blocks() as demo:
    gr.Markdown("## RAG Demo Comparator")
    query_input = gr.Textbox(label="Ask your question:", submit_btn="Submit")
    # submit_button = gr.Button("Submit")
    with gr.Row():
        with gr.Column():
            gr.Markdown('### Simple LLM Response')
            response_1 = gr.Textbox(label="LLM Response", interactive=False)
        with gr.Column():
            gr.Markdown('### Standard RAG Response')
            response_2 = gr.Textbox(label="RAG Response", interactive=False)
            show_cxt_2_btn = gr.Button("Click to see context")
            context_2 = gr.Textbox(visible=False, label="Contexts", lines=20)
            
            show_cxt_2_btn.click(fn=lambda: gr.update(visible=False), outputs=show_cxt_2_btn)
            show_cxt_2_btn.click(fn=lambda: gr.update(visible=True), outputs=context_2)
            

        # with gr.Column():
        
        query_input.submit(
            handle_submit,
            inputs=[query_input],
            outputs=[response_1, response_2, context_2]
        )

demo.launch()