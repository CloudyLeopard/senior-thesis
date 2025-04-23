import gradio as gr

# import essentials
from kruppe.llm import OpenAILLM
from kruppe.algorithm.background import BackgroundResearcher

# import librarian
from block_librarian import get_librarian

bkg_researcher = None

def get_background_researcher():
    """For other blocks to access the global background researcher object"""
    global bkg_researcher
    return bkg_researcher

async def initialize_background_researcher(research_question, param_model, param_num_info_requests, param_verbatim_answer, param_strict_answer):
    global bkg_researcher

    librarian = get_librarian()

    if librarian is None:
        gr.Warning("Librarian not initialized. Please initialize the librarian first.")
        return
    
    if research_question is None or research_question == "":
        gr.Warning("Research question cannot be empty.")
        return

    # initialize the LLM model
    llm = OpenAILLM(model=param_model)

    # initialize the background researcher
    other_configs = {
        "num_info_requests": param_num_info_requests,
        "verbatim_answer": param_verbatim_answer,
        "strict_answer": param_strict_answer
    }
    bkg_researcher = BackgroundResearcher(
        llm=llm,
        librarian=librarian,
        research_question=research_question,
        **other_configs
    )

    return f"BackgroundResearcher initialized. Research question: {bkg_researcher.research_question}"

async def execute_background_researcher():
    global bkg_researcher

    if bkg_researcher is None:
        gr.Warning("BackgroundResearcher not initialized. Please initialize the BackgroundResearcher first.")
        return

    report = await bkg_researcher.execute()

    info_history_str = "\n".join([f"Q: {info_request}\nA: {response.text}\n" for info_request, response in bkg_researcher.info_history])
    return report.text, info_history_str

async def refresh_background_researcher():
    global bkg_researcher

    if bkg_researcher is None:
        gr.Warning("BackgroundResearcher not initialized. Please initialize the BackgroundResearcher first.")
        return "", ""
    
    report = bkg_researcher.latest_report
    if report is None:
        return "No existing background research report found.", ""
    else:
        info_history_str = "\n".join([f"Q: {info_request}\nA: {response.text}\n" for info_request, response in bkg_researcher.info_history])
        return report.text, info_history_str

def create_background_block():
    with gr.Blocks() as block:
        gr.Markdown('# Background Researcher')
        research_question = gr.Textbox("What are the key developments and financial projections for Amazon's advertising business, and how is it positioning itself in the digital ad market?",
                                       label="Research Question", interactive=True)

        gr.Markdown('## Initialize BackgroundResearcher')

        with gr.Group():
            gr.Markdown('### `BackgroundResearcher` Configuration')

            with gr.Row():
                param_model = gr.Dropdown(
                    label="Select LLM Model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    multiselect=False,
                    value="gpt-4o-mini",
                    interactive=True,
                    allow_custom_value=False,
                )

                param_num_info_requests = gr.Number(3, label="Number of Information Requests", minimum=1, maximum=10, step=1)
                param_verbatim_answer = gr.Checkbox(False, label="Verbatim Answer")
                param_strict_answer = gr.Checkbox(True, label="Strict Answer")

        init_button = gr.Button("Initialize BackgroundResearcher", variant='primary')
        output_config = gr.Textbox(label="Initialization Status", interactive=False)

        init_button.click(initialize_background_researcher, inputs=[research_question, param_model, param_num_info_requests, param_verbatim_answer, param_strict_answer], outputs=[output_config])

        gr.Markdown('## Execute BackgroundResearcher')

        with gr.Row():
            execute_button = gr.Button("Execute BackgroundResearcher", variant="huggingface")
            refresh_button = gr.Button("Refresh/Check for Existing Reports")

        with gr.Row():
            output_report = gr.Textbox(label="Background Report", lines=10, interactive=False)
            output_contexts = gr.Textbox(label="Contexts", lines=10, interactive=False)
        
        execute_button.click(execute_background_researcher, outputs=[output_report, output_contexts])
        refresh_button.click(refresh_background_researcher, outputs=[output_report, output_contexts])
    return block