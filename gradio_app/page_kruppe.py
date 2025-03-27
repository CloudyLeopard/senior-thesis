import gradio as gr

from block_librarian import create_librarian_block
from block_background import create_background_block
from block_overseer import create_overseer_block
from block_hypothesis import create_hypothesis_block

librarian_block = create_librarian_block()
background_block = create_background_block()
overseer_block = create_overseer_block()
hypothesis_block = create_hypothesis_block()

def tab1_function(input_text):
    return f"Tab 1 received: {input_text}"

def tab2_function(input_text):
    return f"Tab 2 received: {input_text}"

def tab3_function(input_text):
    return f"Tab 3 received: {input_text}"

def tab4_function(input_text):
    return f"Tab 4 received: {input_text}"

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Librarian"):
            librarian_block.render()
        with gr.Tab("Background Research"):
            background_block.render()
        with gr.Tab("Overseer"):
            overseer_block.render()
        with gr.Tab("Hypothesis Research"):
            hypothesis_block.render()

demo.launch()