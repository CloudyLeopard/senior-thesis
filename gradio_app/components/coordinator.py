import gradio as gr
from typing import get_type_hints

from kruppe.algorithm.librarian import Librarian
from kruppe.llm import OpenAILLM
from kruppe.algorithm.coordinator import Coordinator
from utils.tools import ALL_TOOLS, index, docstore
from kruppe.models import Response, Document

# Initialize Coordinator
coordinator = None

async def initialize_coordinator():
    global docstore, index, ALL_TOOLS, coordinator

    llm = OpenAILLM()

    librarian = Librarian(
        llm=llm,
        docstore=docstore,
        index=index,
        toolkit=list(ALL_TOOLS.values()),
        max_steps=20
    )

    tree_configs = {
        "llm": llm,
        "toolkit": list(ALL_TOOLS.values()),
        "docstore": docstore,
        "index": index,
        "max_step": 20,
        "max_degree": 2
    }

    # Initialize Coordinator with all tools by default
    coordinator = Coordinator(
        llm=llm,
        librarian=librarian,
        tree_configs=tree_configs
    )
    
    return coordinator

# Update configuration function
async def update_configuration(model_name_value, max_steps_value, max_degree_value, *tool_values):
    global coordinator
    
    if coordinator is None:
        return "Coordinator not initialized. Please refresh the page."
    
    try:
        # Update model name
        coordinator.llm.model = model_name_value
        coordinator.tree_configs['llm'].model = model_name_value

        # Update max steps
        coordinator.tree_configs['max_steps'] = max_steps_value

        # Update max degree
        coordinator.tree_configs['max_degree'] = max_degree_value

        # Update toolkit based on selected tools
        selected_tools = []
        for tool_name, is_selected in zip(ALL_TOOLS.keys(), tool_values):
            if is_selected:
                selected_tools.append(ALL_TOOLS[tool_name])
        coordinator.tree_configs['toolkit'] = selected_tools
    
        return "Configuration updated successfully!"
    except Exception as e:
        return f"Error updating configuration: {str(e)}"

# Execute query function
async def execute_query(query, n_experts):
    global coordinator
    
    if coordinator is None:
        return "Please refresh the page to initialize the Coordinator."
    
    try:
        # Execute coordinator research
        rt_responses = await coordinator.execute(query, n_experts)
        
        choices = []

        # Format sources
        for i in range(len(rt_responses)):
            response = rt_responses[i]
            sources_html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
            for source in rt_responses[0].sources:
                sources_html += f"""
                    <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
                        <strong>Title:</strong> <a href="{source.metadata.get('url', '#')}">{source.metadata.get("title")}</a><br>
                        <strong>Source:</strong> {source.metadata.get("datasource")}<br>
                        <strong>Text:</strong> {source.text[:500]}...<br>
                    </div>
                """
            sources_html += "</div>"

            entire_text = response.text + "##########" + sources_html
            choices.append((f"{response.metadata['expert']}: {response.text[:100]}", entire_text))
        
        return gr.update(choices=choices, value=None)
        
    except Exception as e:
        return f"Error executing coordinator research: {str(e)}"

async def populate_report(dropdown_value):
    if dropdown_value is None:
        return "", ""
    
    # Split the dropdown value to get the response text and HTML
    response_text, html = dropdown_value.split("##########")

    return response_text, html

async def show_research_forest():
    """Handler function for the visualize forest button"""
    global coordinator
    if coordinator is None:
        return "Please initialize the coordinator first."
    return coordinator.visualize_research_forest()

def create_coordinator_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Coordinator Interface")
        
        # Parameter Configuration Section (Collapsible)
        with gr.Accordion("Coordinator Configuration", open=False) as config_accordion:
            with gr.Row():
                with gr.Column():
                    # Model Configuration
                    gr.Markdown("### Model Configuration")
                    model_name = gr.Dropdown(
                        choices=[
                            "gpt-4o",
                            "gpt-4o-mini", 
                            "gpt-4.1",
                            "gpt-4.1-mini",
                            "gpt-4.1-nano",
                            "gpt-4.5-preview"
                        ],
                        value="gpt-4.1-mini",
                        label="Model Name",
                        interactive=True
                    )
                    
                    # Max Steps Configuration
                    gr.Markdown("### Execution Configuration")
                    max_steps = gr.Number(
                        value=20,
                        label="Maximum Steps",
                        minimum=1,
                        maximum=100,
                        step=1
                    )
                    
                    # Max Degree Configuration
                    max_degree = gr.Number(
                        value=2,
                        label="Maximum Degree",
                        minimum=1,
                        maximum=5,
                        step=1
                    )

                    # Toolkit Configuration
                    gr.Markdown("### Available Tools")
                    tool_checkboxes = []
                    for tool_name in ALL_TOOLS.keys():
                        tool_checkboxes.append(
                            gr.Checkbox(
                                label=tool_name,
                                value=True,  # All tools enabled by default
                                interactive=True
                            )
                        )
                    
                    # Update Button
                    update_btn = gr.Button("Update Configuration", variant="primary")
                    update_status = gr.Textbox(label="Update Status", interactive=False)
        
        # Main Query Interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Query Configuration")
                n_experts_input = gr.Number(
                    value=3,
                    label="N Expert Perspectives to Generate",
                    minimum=1,
                    maximum=10,
                    step=1
                )
            with gr.Column(scale=3):
                gr.Markdown("### Query Input")
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="Enter your query here...",
                    lines=1,
                    submit_btn=True
                )

        gr.Markdown("## Research Report")

        report_dropdown = gr.Dropdown(
            choices=[],
            label="Select Research Report (Once it's generated)",
            interactive=True
        )
        
        with gr.Group():
            research_output = gr.Textbox(
                label="Research Report",
                lines=5,
                max_lines=20,
                interactive=False
            )
            
            with gr.Accordion("Sources", open=False):
                sources_output = gr.HTML(label="Sources")
        
        # Add visualization button and output
        with gr.Group():
            visualize_btn = gr.Button("Visualize Research Forest", variant="secondary")
            forest_visualization = gr.HTML(label="Research Forest Structure")
        
        # Set up event handlers
        update_btn.click(
            update_configuration,
            inputs=[model_name, max_steps, max_degree] + tool_checkboxes,
            outputs=[update_status],
            queue=False
        )
        
        query_input.submit(
            fn=execute_query,
            inputs=[query_input, n_experts_input],
            outputs=[report_dropdown],
            queue=False
        )

        report_dropdown.change(
            fn=populate_report,
            inputs=[report_dropdown],
            outputs=[research_output, sources_output],
            queue=False
        )
        
        # Add event handler for visualization button
        visualize_btn.click(
            fn=show_research_forest,
            outputs=[forest_visualization],
            queue=False
        )
        
    return interface

# Initialize the coordinator when the module is imported
async def init():
    global coordinator
    coordinator = await initialize_coordinator()

# Run the initialization
import asyncio
asyncio.run(init())

