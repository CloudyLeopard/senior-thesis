import gradio as gr
import asyncio


async def ask(message) -> str:
    return f"Response to: {message}"

# Chatbot logic
async def chatbot_logic(message, chat_history, answer_using_story, story_history):
    if answer_using_story:
        # Generate stories and answer using one of them
        stories = ["story 1", "story 2", "story 3"]
        story_history.extend(stories)
        bot_message = f"Answer using story: {stories[0]}"
    else:
        # Perform a simple RAG query
        bot_message = await ask(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return chat_history, story_history

def chatbot_logic_main(message, chat_history, answer_using_story, story_history):
    return asyncio.run(chatbot_logic(message, chat_history, answer_using_story, story_history))

# Gradio UI
def file_upload_handler(file):
    return f"File {file.name} uploaded successfully!"

with gr.Blocks() as demo:
    stories = gr.State([])

    with gr.Row():
        gr.File(label="Upload File", file_types=[".txt", ".csv", ".json", ".pdf"])

    with gr.Row():
        with gr.Column():
            @gr.render(inputs=stories)
            def render_stories(story_list):
                for story in story_list:
                    gr.Textbox(story, show_label=False, container=False)

        with gr.Column():
            chatbot = gr.Chatbot(label="FinRAG Bot", type='messages')
            user_input = gr.Textbox(label="Enter your query")
            answer_using_story_toggle = gr.Checkbox(label="Answer using story")
            submit_button = gr.Button("Submit")

            def handle_query(query, chat_history, answer_using_story, story_history):
                chat_history, generated_stories = chatbot_logic_main(query, chat_history, answer_using_story, story_history)
                return chat_history, generated_stories

            submit_button.click(
                handle_query, 
                inputs=[user_input, chatbot, answer_using_story_toggle, stories], 
                outputs=[chatbot, stories]
            )

# Run the app
demo.launch()


# async def ask(message: str, history: list) -> str:

#     documents = DirectoryData(input_files=files).fetch()
#     embedding_model = OpenAIEmbeddingModel()
#     vector_storage = NumPyVectorStorage(embedding_model=embedding_model)
#     text_splitter = RecursiveTextSplitter()
#     chunked_documents = text_splitter.split_documents(documents)
#     await vector_storage.async_insert_documents(chunked_documents)

#     query = message
#     relevant_documents = await vector_storage.async_similarity_search(query, top_k=5)

#     llm = OpenAILLM("gpt-4o")

#     prompt_formatter = RAGPromptFormatter()
#     prompt_formatter.add_documents(relevant_documents)
#     messages = prompt_formatter.format_messages(user_prompt=query)

#     response = await llm.async_generate(messages)

#     return response