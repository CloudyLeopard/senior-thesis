import gradio as gr
import asyncio
from typing import Tuple, List
import json

from kruppe.prompt_formatter import SimplePromptFormatter, RAGPromptFormatter
from kruppe.llm import OpenAILLM
from kruppe.vector_storages import NumPyVectorStorage
from kruppe.embeddings import OpenAIEmbeddingModel
from kruppe.tools.searcher import NewsArticleSearcher
from kruppe.tools.sources import FinancialTimesData

llm = OpenAILLM()
with open(".ft-headers.json") as f:
    ft_headers = json.load(f)

searcher = NewsArticleSearcher(sources=[FinancialTimesData(headers=ft_headers)])

async def ask(message) -> str:
    formatter = SimplePromptFormatter()
    messages = formatter.format_messages(user_prompt=message)
    response = await llm.async_generate(messages)
    return response

async def build_vectorstore(message):
    # internet search documents
    vectorstore = NumPyVectorStorage(embedding_model=OpenAIEmbeddingModel())    
    searched_documents = await searcher.async_search(query=message)
    await vectorstore.async_insert_documents(searched_documents)

    return vectorstore

async def rag_ask(message, vectorstore) -> Tuple[str, List]:
    # retrieve and rag format
    relevant_documents = await vectorstore.async_similarity_search(message, top_k=5)
    prompt_formatter = RAGPromptFormatter()
    prompt_formatter.add_documents(relevant_documents)
    messages = prompt_formatter.format_messages(user_prompt=message)
    response = await llm.async_generate(messages)
    contexts = [doc.text for doc in relevant_documents]
    return (response, contexts)


# Chatbot logic
async def chatbot_logic(message, chat_history, answer_using_rag, vectorstore):
    retrieved_contexts = []
    if answer_using_rag and vectorstore is None:
        raise gr.Error("Please build vectorstore first.")
    
    if answer_using_rag:
        bot_message, retrieved_contexts = await rag_ask(message, vectorstore)
    else:
        # Perform a simple RAG query
        bot_message = await ask(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return chat_history, retrieved_contexts

def chatbot_logic_main(message, chat_history, answer_using_rag, files):
    return asyncio.run(chatbot_logic(message, chat_history, answer_using_rag, files))


with gr.Blocks() as demo:
    contexts = gr.State([])
    vectorstore = gr.State()

    # with gr.Row():
    #     files = gr.File(label="Upload File", file_count = "multiple", file_types=[".txt", ".pdf"])

    with gr.Row():
        with gr.Column():
            @gr.render(inputs=contexts)
            def render_stories(context_list):
                gr.Textbox("Contexts used for RAG")
                for context in context_list:
                    gr.Textbox(context, show_label=False, container=False)

        with gr.Column():
            chatbot = gr.Chatbot(label="FinRAG Bot", type='messages')
            user_input = gr.Textbox(label="Enter your query to ask chatbot or build vectorstore. Model is gpt-4o-mini")
            answer_using_rag_toggle = gr.Checkbox(label="Answer using RAG. Please build vectorstore first.")
            submit_button = gr.Button("Ask chatbot")
            vectorstore_button = gr.Button("Search FT and Build vectorstore. This may take a while...")
            loading_indicator = gr.Label(value="Vectorstore not built yet. Enter a query and select \"Build vectorstore\".", label="Vectorstore status")

            def handle_query(query, chat_history, answer_using_rag, vectorstore):
                chat_history, generated_contexts = chatbot_logic_main(query, chat_history, answer_using_rag, vectorstore)
                return chat_history, generated_contexts
            
            def build_vectorstore_with_loading(input_text):
                # Start loading
                yield "Processing...", None
                result = asyncio.run(build_vectorstore(input_text))
                # Stop loading and display result
                yield f"Vectorstore ready! Contains {len(result.documents)} documents.",  result

            vectorstore_button.click(
                build_vectorstore_with_loading, 
                inputs=[user_input], 
                outputs=[loading_indicator, vectorstore]
            )

            submit_button.click(
                handle_query, 
                inputs=[user_input, chatbot, answer_using_rag_toggle, vectorstore], 
                outputs=[chatbot, contexts]
            )

# Run the app
demo.launch(share=True, auth=("test", "123456"))