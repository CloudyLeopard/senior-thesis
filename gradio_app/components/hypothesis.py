import gradio as gr
from typing import get_type_hints

from kruppe.llm import OpenAILLM
from kruppe.algorithm.hypothesis import HypothesisResearcher
from utils.tools import ALL_TOOLS, index, docstore
from kruppe.models import Response, Document

# Initialize Hypothesis Researcher
research_tree = None

async def initialize_hypothesis_explorer():
    global docstore, index, ALL_TOOLS, research_tree

    # Initialize Hypothesis Researcher with all tools by default
    research_tree = HypothesisResearcher(
        llm=OpenAILLM(model="gpt-4.1-mini"),
        docstore=docstore,
        index=index,
        toolkit=list(ALL_TOOLS.values()),
        max_steps=20,
        max_degree=2,
        role="Financial Analyst",
        role_description="A financial analyst is someone who is great at analyzing finance stuff",
        background_report=""
    )
    
    return research_tree

# Update configuration function
async def update_configuration(model_name_value, max_steps_value, max_degree_value, role_value, role_description_value, background_report_value, *tool_values):
    global research_tree
    
    if research_tree is None:
        return "Hypothesis Researcher not initialized. Please refresh the page."
    
    try:
        # Update model name
        research_tree.llm.model = model_name_value

        # Update max steps
        research_tree.max_steps = max_steps_value
        
        # Update max degree
        research_tree.max_degree = max_degree_value
        
        # Update role and description
        research_tree.role = role_value
        research_tree.role_description = role_description_value
        
        # Update background report
        research_tree.background_report = background_report_value

        # Update toolkit based on selected tools
        selected_tools = []
        for tool_name, is_selected in zip(ALL_TOOLS.keys(), tool_values):
            if is_selected:
                selected_tools.append(ALL_TOOLS[tool_name])
        research_tree.toolkit = selected_tools
        
        return "Configuration updated successfully!"
    except Exception as e:
        return f"Error updating configuration: {str(e)}"

# Execute query function
async def execute_query(query):
    global research_tree
    
    if research_tree is None:
        return "Please refresh the page to initialize the Hypothesis Researcher."
    
    try:
        # Execute hypothesis research
        rt_responses = await research_tree.execute(query)
        # rt_responses = [
        #     Response(text="abc123", sources=[Document(text="blah", metadata={})]),
        #     Response(text="abc123", sources=[Document(text="blah", metadata={})]),
        #     Response(text="abc123", sources=[Document(text="blah", metadata={})]),
        # ]
        
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
            choices.append((response.text[:100], entire_text))
        
        return gr.update(choices=choices, value=None)
        
    except Exception as e:
        return f"Error executing hypothesis research: {str(e)}"

async def populate_report(dropdown_value):
    if dropdown_value is None:
        return "", ""
    
    # Split the dropdown value to get the response text and HTML
    response_text, html = dropdown_value.split("##########")

    return response_text, html

def create_hypothesis_interface():
    with gr.Blocks() as interface:
        

        gr.Markdown("# Hypothesis Researcher Interface")
        
        # Parameter Configuration Section (Collapsible)
        with gr.Accordion("Hypothesis Researcher Configuration", open=False) as config_accordion:
            with gr.Row():
                with gr.Column():
                    # Model Configuration
                    gr.Markdown("### Model")
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
                    gr.Markdown("### Max Steps")
                    max_steps = gr.Number(
                        value=20,
                        label="Maximum Steps",
                        minimum=1,
                        maximum=100,
                        step=1
                    )
                    
                    # Max Degree Configuration
                    gr.Markdown("### Max Degree")
                    max_degree = gr.Number(
                        value=2,
                        label="Maximum Degree",
                        minimum=1,
                        maximum=5,
                        step=1
                    )
                    
                    # Role Configuration
                    gr.Markdown("### Role")
                    role = gr.Textbox(
                        value="Financial Analyst",
                        label="Role",
                        interactive=True
                    )
                    
                    # Role Description
                    gr.Markdown("### Role Description")
                    role_description = gr.Textbox(
                        value="A financial analyst is someone who is great at analyzing finance stuff",
                        label="Role Description",
                        lines=3,
                        interactive=True
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
                    
                    # Background Report
                    gr.Markdown("### Background Report")
                    background_report = gr.Textbox(
                        value="",
                        label="Background Report",
                        lines=5,
                        placeholder="Enter any background information or context for the research...",
                        interactive=True
                    )
                    
                    # Update Button
                    update_btn = gr.Button("Update Configuration", variant="primary")
                    update_status = gr.Textbox(label="Update Status", interactive=False)
        
        # Main Query Interface
        query_input = gr.Textbox(
            label="Query",
            placeholder="Enter your query here...",
            lines=1,
            submit_btn=True
        )

        gr.Markdown("### Research Report")

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
        
        # Set up event handlers
        update_btn.click(
            update_configuration,
            inputs=[model_name, max_steps, max_degree, role, role_description, background_report] + tool_checkboxes,
            outputs=[update_status],
            queue=False
        )
        
        query_input.submit(
            fn=execute_query,
            inputs=[query_input],
            outputs=[report_dropdown],
            queue=False
        )

        report_dropdown.change(
            fn=populate_report,
            inputs=[report_dropdown],
            outputs=[research_output, sources_output],
            queue=False
        )

        
    return interface

# Initialize the hypothesis researcher when the module is imported
async def init():
    global research_tree
    research_tree = await initialize_hypothesis_explorer()

# Run the initialization
import asyncio
asyncio.run(init()) 