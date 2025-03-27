import gradio as gr
from pytest import param

# import essentials
from kruppe.llm import OpenAILLM
from kruppe.algorithm.overseer import Overseer

# import librarian and background researcher
from block_librarian import get_librarian
from block_background import get_background_researcher

# global states
overseer = None

async def initialize_overseer(research_question, param_model, param_num_leads, param_hyp_model, param_iterations, param_iterations_used, param_num_info_requests, param_verbatim_answer, param_strict_answer):
    global overseer

    librarian = get_librarian()
    background_researcher = get_background_researcher()

    if librarian is None:
        gr.Warning("Librarian not initialized. Please initialize the librarian first.")
        return
    if background_researcher is None:
        gr.Warning("BackgroundResearcher not initialized. Please initialize the BackgroundResearcher first.")
        return
    if research_question is None or research_question == "":
        gr.Warning("Research question cannot be empty.")
        return
    
    # initialize the LLM model
    overseer_llm = OpenAILLM(model=param_model)
    overseer_hyp_llm = OpenAILLM(model=param_hyp_model)

    # initialize the overseer
    other_configs = {
        "num_leads": param_num_leads,
        "hyp_researcher_config": {
            "llm": overseer_hyp_llm,
            "iterations": param_iterations,
            "iterations_used": param_iterations_used,
            "num_info_requests": param_num_info_requests,
            "verbatim_answer": param_verbatim_answer,
            "strict_answer": param_strict_answer
        }
    }
    overseer = Overseer(
        llm=overseer_llm,
        research_question=research_question,
        librarian=librarian,
        bkg_researcher=background_researcher,
        **other_configs
    )

    return "Overseer initialized. " + ("Background report found" if overseer.background_report else "Background report not created yet")

def create_overseer_block():
    with gr.Blocks() as block:
        gr.Markdown('# Overseer')
        research_question = gr.Textbox("What are the key developments and financial projections for Amazon's advertising business, and how is it positioning itself in the digital ad market?",
                                       label="Research Question", interactive=True)
        
        gr.Markdown('## Initialize `Overseer`')
        with gr.Group():
            gr.Markdown('### `Overseer` Configurations')

            with gr.Row():
                param_model = gr.Dropdown(
                    label="Select LLM Model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    multiselect=False,
                    value="gpt-4o-mini",
                    interactive=True,
                    allow_custom_value=False,
                )

                param_num_leads = gr.Number(3, label="Number of Leads", minimum=1, maximum=10, step=1)

            gr.Markdown('#### `HypothesisResearcher` Configurations')
            with gr.Row():
                param_hyp_model = gr.Dropdown(
                    label="Select LLM Model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    multiselect=False,
                    value="gpt-4o-mini",
                    interactive=True,
                    allow_custom_value=False,
                )

                param_iterations = gr.Number(3, label="Number of Iterations", minimum=1, maximum=10, step=1)
                param_iterations_used = gr.Number(1, label="Number of Iterations Used", minimum=1, maximum=5, step=1)

            with gr.Row():    
                param_num_info_requests = gr.Number(3, label="Number of Information Requests", minimum=1, maximum=10, step=1)
                param_verbatim_answer = gr.Checkbox(False, label="Verbatim Answer")
                param_strict_answer = gr.Checkbox(True, label="Strict Answer")
            
            init_button = gr.Button("Initialize Overseer")
            output_config = gr.Textbox(label="Initialization Status", interactive=False)

            init_button.click(initialize_overseer, inputs=[research_question, param_model, param_num_leads, param_hyp_model, param_iterations, param_iterations_used, param_num_info_requests, param_verbatim_answer, param_strict_answer], outputs=output_config)

    return block