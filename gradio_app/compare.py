import gradio as gr
import asyncio

from rag.operate import ask_simple_llm, ask_simple_rag, build_vectorstore_index
from rag.llm import OpenAIEmbeddingModel, OpenAILLM

embedding_model=OpenAIEmbeddingModel()
llm = OpenAILLM()
# index = asyncio.run(build_vectorstore_index(embedding_model=embedding_model))
index = None

def handle_submit(query):
    llm_response, simple_rag_response = asyncio.run(asyncio.gather(
        ask_simple_llm(query, llm),
        ask_simple_rag(query, index, embedding_model, llm)
    ))

    return (llm_response,
            simple_rag_response["response"], simple_rag_response["contexts"],
            )

with gr.Blocks() as demo:
    gr.Markdown("### RAG Demo Comparator")
    with gr.Row():
        query_input = gr.Textbox(label="Ask your question:")
        submit_button = gr.Button("Submit")
    with gr.Row():
        with gr.Column():
            response_1 = gr.Textbox(label="LLM Response", interactive=False)
        with gr.Column():
            response_2 = gr.Textbox(label="RAG Response", interactive=False)
            dropdown_2 = gr.Dropdown(label="Show Contexts", interactive=False)

        # with gr.Column():
        
        submit_button.click(
            handle_submit,
            inputs=[query_input],
            outputs=[]
        )