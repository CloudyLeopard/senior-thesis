import gradio as gr

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

async def create_leads():
    global overseer

    if overseer is None:
        gr.Warning("Overseer not initialized. Please initialize the Overseer first.")
        return
    
    leads = await overseer.create_leads()
    if len(leads) > 3:
        gr.Warning("More than 3 leads generated. Only the first 3 leads are displayed.")
        leads = leads[:3]
    if len(leads) < 3:
        leads = leads + [None] * (3 - len(leads))
    
    def format_lead(lead):
        if lead is None:
            return ""
        else:
            return (f"**Lead:** {lead.lead}\n"
                    + f"**Hypothesis:** {lead.hypothesis}\n"
                    + f"**Observation:** {lead.observation}")
    
    return format_lead(leads[0]), format_lead(leads[1]), format_lead(leads[2])

async def execute_overseer():
    global overseer

    if overseer is None:
        gr.Warning("Overseer not initialized. Please initialize the Overseer first.")
        return
    
    results = await overseer.execute()

    if len(results) > 3:
        gr.Warning("More than 3 reports generated. Only the first 3 reports are displayed.")
        results = results[:3]
    if len(results) < 3:
        results = results + [None] * (3 - len(results))
    
    def format_report(result):
        if result is None or result["report"] is None:
            return ""
        else:
            return (f"**Original Lead:** {result['original_lead'].lead}\n\n" 
                    + f"**Hypothesis:** {result['hypothesis']}\n\n"
                    + f"---Report---\n{result['report'].text}")
    
    return format_report(results[0]), format_report(results[1]), format_report(results[2])    

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
            
            init_button = gr.Button("Initialize Overseer", variant='primary')
            output_config = gr.Textbox(label="Initialization Status", interactive=False)

            init_button.click(initialize_overseer, inputs=[research_question, param_model, param_num_leads, param_hyp_model, param_iterations, param_iterations_used, param_num_info_requests, param_verbatim_answer, param_strict_answer], outputs=output_config)

        # ------------------------------------------------

        gr.Markdown('## Creating Leads (Testing)')

        create_leads_button = gr.Button("Create Leads", variant='huggingface')
        with gr.Row():
            output_lead_1 = gr.Textbox(label="Lead 1", lines=10, interactive=False)
            output_lead_2 = gr.Textbox(label="Lead 2", lines=10, interactive=False)
            output_lead_3 = gr.Textbox(label="Lead 3", lines=10, interactive=False)
        
        create_leads_button.click(create_leads, outputs=[output_lead_1, output_lead_2, output_lead_3])

        # ------------------------------------------------

        gr.Markdown('## Execute Overseer\n*Warning: This may take a while.*')

        execute_button = gr.Button("Execute Overseer", variant='huggingface')

        with gr.Row():
            output_1 = gr.Textbox(label="Report 1", lines=10, interactive=False)
            output_2 = gr.Textbox(label="Report 2", lines=10, interactive=False)
            output_3 = gr.Textbox(label="Report 3", lines=10, interactive=False)

        execute_button.click(execute_overseer, outputs=[output_1, output_2, output_3])
    return block