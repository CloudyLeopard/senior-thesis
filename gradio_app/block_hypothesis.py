import gradio as gr

# import essentials
from kruppe.llm import OpenAILLM
from kruppe.algorithm.hypothesis import HypothesisResearcher
from kruppe.algorithm.agents import Lead

# import librarian
from block_librarian import get_librarian

# global states
hyp_researcher = None

async def initialize_hypothesis_researcher(research_question, lead_observation, lead_lead, lead_hypothesis, param_hyp_model, param_iterations, param_iterations_used, param_num_info_requests, param_verbatim_answer, param_strict_answer):
    global hyp_researcher

    librarian = get_librarian()

    if librarian is None:
        gr.Warning("Librarian not initialized. Please initialize the librarian first.")
        return
    
    if research_question is None or research_question == "":
        gr.Warning("Research question cannot be empty.")
        return
    
    if not all([lead_observation, lead_lead, lead_hypothesis]):
        gr.Warning("Lead fields cannot be empty.")
        return
    
    lead = Lead(
        observation=lead_observation,
        lead=lead_lead,
        hypothesis=lead_hypothesis
    )

    # initialize the LLM model
    llm = OpenAILLM(model=param_hyp_model)

    # initialize the hypothesis researcher
    other_configs = {
        "iterations": param_iterations,
        "iterations_used": param_iterations_used,
        "num_info_requests": param_num_info_requests,
        "verbatim_answer": param_verbatim_answer,
        "strict_answer": param_strict_answer
    }
    hyp_researcher = HypothesisResearcher(
        llm=llm,
        librarian=librarian,
        new_lead=lead,
        research_question=research_question,
        **other_configs
    )

    return f"HypothesisResearcher initialized. Current research lead: {hyp_researcher.new_lead.lead}"

async def execute_hypothesis():
    global hyp_researcher

    if hyp_researcher is None:
        gr.Warning("HypothesisResearcher not initialized. Please initialize the HypothesisResearcher first.")
        return

    report = await hyp_researcher.execute()

    return hyp_researcher.latest_report, hyp_researcher.latest_hypothesis

def create_hypothesis_block():
    with gr.Blocks() as block:
        gr.Markdown('# Hypothesis Researcher')

        with gr.Group():
            research_question = gr.Textbox("What are the key developments and financial projections for Amazon's advertising business, and how is it positioning itself in the digital ad market?",
                                       label="Research Question", interactive=True)
            gr.Markdown('### Customize `Lead`')
            with gr.Row():
                lead_observation = gr.Textbox("Amazon's advertising revenue is projected to grow to $55 billion by 2025, yet it currently significantly trails Google and Meta's advertising revenues, estimated at $250 billion and $130 billion respectively in 2023. This wide gap persists despite Amazon's rapid growth and unique data advantages.",
                                        label="Lead Observation", lines=5, interactive=True)
                lead_lead = gr.Textbox("Investigate the strategies Amazon might employ to scale its advertising revenue more rapidly to close the gap with Google and Meta. Particularly explore how Amazon can leverage its e-commerce platform to create new advertising models that competitors cannot replicate.",
                                        label="Lead",lines=5, interactive=True)
                lead_hypothesis = gr.Textbox("Amazon will develop innovative advertising formats integrated within its e-commerce transactions that will not only boost engagement but also drive higher conversion rates and revenue, positioning it uniquely against Google and Meta's traditional ad offerings.",
                                       label="Hypothesis", lines=5, interactive=True)
        
        gr.Markdown('## Initialize HypothesisResearcher')

        with gr.Group():
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
            
            init_button = gr.Button("Initialize HypothesisResearcher", variant='primary')
            output_config = gr.Textbox(label="Initialization Status", interactive=False)

            init_button.click(initialize_hypothesis_researcher, inputs=[research_question, lead_observation, lead_lead, lead_hypothesis, param_hyp_model, param_iterations, param_iterations_used, param_num_info_requests, param_verbatim_answer, param_strict_answer], outputs=output_config)

        # ------------------------------------------------
    
        gr.Markdown('## HypothesisResearcher Execution')

        execute_button = gr.Button("Execute HypothesisResearcher", variant='huggingface')
        output_hypothesis = gr.Textbox(label="Hypothesis", interactive=False)
        output_report = gr.Textbox(label="Research Report", lines=10, interactive=False)

        execute_button.click(execute_hypothesis, outputs=[output_report, output_hypothesis])

    return block