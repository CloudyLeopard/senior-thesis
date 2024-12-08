import gradio as gr
import asyncio

from rag.tools.sources import DirectoryData
from rag.embeddings import OpenAIEmbeddingModel
from rag.text_splitters import RecursiveTextSplitter
from rag.vector_storages import NumPyVectorStorage
from rag.llm import OpenAILLM
from rag.prompts import RAGPromptFormatter

async def predict(message, history) -> str:
    files = []
    for msg in history:
        if msg['role'] == "user" and isinstance(msg['content'], tuple):
            files.append(msg['content'][0])
    for file in message["files"]:
        files.append(file)
    
    documents = DirectoryData(input_files=files).fetch()
    embedding_model = OpenAIEmbeddingModel()
    vector_storage = NumPyVectorStorage(embedding_model=embedding_model)
    text_splitter = RecursiveTextSplitter()
    chunked_documents = text_splitter.split_documents(documents)
    await vector_storage.async_insert_documents(chunked_documents)

    query = message["text"]
    relevant_documents = await vector_storage.async_similarity_search(query, top_k=5)

    llm = OpenAILLM("gpt-4o")

    prompt_formatter = RAGPromptFormatter()
    prompt_formatter.add_documents(relevant_documents)
    messages = prompt_formatter.format_messages(user_prompt=query)

    response = await llm.async_generate(messages)

    return response

def predict_main(message, history):
    return asyncio.run(predict(message, history))

demo = gr.ChatInterface(
    predict_main,
    type="messages",
    title="Fin RAG Chatbot",
    description="Upload any text or pdf files as \"priority\" sources, and ask financial queries",
    textbox = gr.MultimodalTextbox(file_count="multiple", file_types=[".txt", ".pdf"]),
    multimodal=True
)

demo.launch()