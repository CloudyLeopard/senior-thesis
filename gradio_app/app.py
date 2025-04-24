import gradio as gr
from components.coordinator import create_coordinator_interface
from components.hypothesis import create_hypothesis_interface
from components.librarian import create_librarian_interface

coordinator_interface = create_coordinator_interface()
hypothesis_interface = create_hypothesis_interface()
librarian_interface = create_librarian_interface()

with gr.Blocks(title="Research Assistant") as demo:
    with gr.Tabs():
        with gr.Tab("Coordinator"):
            coordinator_interface.render()
        
        with gr.Tab("Hypothesis Researcher"):
            hypothesis_interface.render()
        
        with gr.Tab("Librarian"):
            librarian_interface.render()

demo.launch()