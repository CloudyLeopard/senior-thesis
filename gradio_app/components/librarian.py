from typing import get_type_hints

import gradio as gr
from kruppe.algorithm.librarian import Librarian
from kruppe.llm import OpenAILLM
from utils.tools import ALL_TOOLS, index, docstore

# Initialize Librarian immediately
async def initialize_librarian():
    global docstore, index, ALL_TOOLS

    # Initialize Librarian with all tools by default
    librarian = Librarian(
        llm=OpenAILLM(model="gpt-4.1-mini"),
        docstore=docstore,
        index=index,
        toolkit=list(ALL_TOOLS.values()),
        max_steps=20
    )
    
    return librarian

# Update configuration function
async def update_configuration(model_name_value, max_steps_value, *tool_values):
    global librarian
    
    if librarian is None:
        return "Librarian not initialized. Please refresh the page."
    
    try:
        # update model

        # update model name
        librarian.llm.model = model_name_value

        # Update max steps
        librarian.max_steps = max_steps_value
        
        # Update toolkit based on selected tools
        selected_tools = []
        for tool_name, is_selected in zip(ALL_TOOLS.keys(), tool_values):
            if is_selected:
                selected_tools.append(ALL_TOOLS[tool_name])
        librarian.toolkit = selected_tools
        
        return "Configuration updated successfully!"
    except Exception as e:
        return f"Error updating configuration: {str(e)}"

# Execute query function
async def execute_query(query):
    global librarian
    
    if librarian is None:
        return "Please refresh the page to initialize the Librarian.", "", ""
    
    try:
        # Execute query
        response = await librarian.execute(query)
        
        # Format sources
        sources_html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
        for source in response.sources:
            sources_html += f"""
                <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>
                    <strong>Title:</strong> <a href="{source.metadata.get('url', '#')}">{source.metadata.get("title")}</a><br>
                    <strong>Source:</strong> {source.metadata.get("datasource")}<br>
                    <strong>Text:</strong> {source.text[:500]}...<br>
                </div>
            """
        sources_html += "</div>"
        
        # Return results
        return response.text, sources_html
    except Exception as e:
        return f"Error executing query: {str(e)}", "", ""


# Initialize the librarian
librarian = None

def create_librarian_interface():
    
    with gr.Blocks() as interface:
        gr.Markdown("# Librarian Interface")
        
        # Parameter Configuration Section (Collapsible)
        with gr.Accordion("Librarian Configuration", open=False) as config_accordion:
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
        search_query = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query here...",
            lines=1,
            submit_btn=True
        )
        # execute_btn = gr.Button("Execute", variant="primary")

            
        with gr.Group():
            # Final Response Display
            # gr.Markdown("### Response")
            response_output = gr.Textbox(
                label="Response",
                lines=5,
                max_lines=20,
                interactive=False
            )
            
            # Sources Display (Collapsible)
            with gr.Accordion("Sources", open=False) as sources_accordion:
                sources_output = gr.HTML(label="Sources")
        
    
        # Set up event handlers
        update_btn.click(
            update_configuration,
            inputs=[model_name, max_steps] + tool_checkboxes,
            outputs=[update_status],
            queue=False
        )
        
        search_query.submit(
            execute_query,
            inputs=[search_query],
            outputs=[response_output, sources_output],
            queue=False
        )
    
    return interface

# Initialize the librarian when the module is imported
async def init():
    global librarian
    librarian = await initialize_librarian()

# Run the initialization
import asyncio
asyncio.run(init()) 

