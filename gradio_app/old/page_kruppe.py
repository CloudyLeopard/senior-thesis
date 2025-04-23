import gradio as gr

from block_librarian import create_librarian_block
from block_background import create_background_block
from block_coordinator import create_coordinator_block
from block_hypothesis import create_hypothesis_block
from stream_logs import create_log_interface

librarian_block = create_librarian_block()
background_block = create_background_block()
overseer_block = create_coordinator_block()
hypothesis_block = create_hypothesis_block()
log_iface = create_log_interface()

with gr.Blocks() as demo:
    with gr.Sidebar(position="right", width=400):
        log_iface.render()

    with gr.Tabs():
        with gr.Tab("Librarian"):
            librarian_block.render()
        # with gr.Tab("Background Research"):
        #     background_block.render()
        with gr.Tab("Coordinator"):
            overseer_block.render()
        with gr.Tab("Hypothesis Research"):
            hypothesis_block.render()
        # with gr.Tab("Log"):
    

demo.launch()