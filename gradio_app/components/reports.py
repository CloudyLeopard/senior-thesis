import gradio as gr
import pandas as pd
import json
import os

# --- Load your questions DataFrame ---
# Replace with your actual DataFrame loading
df = pd.read_csv("/Users/danielliu/Workspace/fin-rag/experiments/reports.csv")  # should have a column "question"

questions = df["question"].tolist()

def load_reports_for_question(question):
    """
    1) Find the row index i for the chosen question
    2) Load reports_i.json, get its "research_reports" list
    3) Return [(snippet, serialized_report_dict), ...] for the dropdown,
       plus clear both outputs.
    """
    idx_list = df.index[df["question"] == question].tolist()
    if not idx_list:
        return [], "", ""
    i = idx_list[0]

    with open(f"/Users/danielliu/Workspace/fin-rag/experiments/kruppe_report/report_{i}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    options = []
    for r in data.get("research_reports", []):
        txt = r.get("text", "")
        snippet = txt[:100].replace("\n", " ")
        if len(txt) > 100:
            snippet += "…"
        # store the entire report dict as a JSON string in the value
        options.append((snippet, json.dumps(r)))
    # clear the report text + sources HTML
    return gr.update(choices=options, value=None), "", "", str(i)

def display_report_and_sources(report_json):
    """
    Parse the selected report JSON, extract `text` and `sources`,
    build HTML for each source, and return (full_text, html_sources).
    """
    if not report_json:
        return "", ""
    r = json.loads(report_json)
    full_text = r.get("text", "")

    # Build sources HTML
    sources = r.get("sources", [])
    html = "<div style='display:flex; flex-direction:column; gap:10px;'>"
    for src in sources:
        meta = src.get("metadata", {})
        url  = meta.get("url", "#")
        title= meta.get("title", "No title")
        ds   = meta.get("datasource", "Unknown")
        txt  = src.get("text", "")
        snippet = (txt[:500] + "...") if len(txt) > 500 else txt
        html += f"""
          <div style='border:1px solid #ccc; padding:10px; border-radius:5px;'>
            <strong>Title:</strong> <a href="{url}">{title}</a><br>
            <strong>Source:</strong> {ds}<br>
            <strong>Text:</strong> {snippet}
          </div>
        """
    html += "</div>"

    return full_text, html

def load_forest_html(i):
    """
    Load the visualization HTML for index i from forest_{i}.html.
    """
    if i is None:
        return "<p style='color:red;'>No question selected yet.</p>"
    path = f"/Users/danielliu/Workspace/fin-rag/experiments/kruppe_report/report_{i}_forest.html"
    if not os.path.exists(path):
        return f"<p style='color:red;'>File {path} not found.</p>"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_report_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## Question → Research Report Explorer")

        # Question selector
        question_dd = gr.Dropdown(
            choices=questions,
            label="Select a question",
            interactive=True
        )

        # Report snippet selector (initially empty)
        report_dd = gr.Dropdown(
            choices=[],
            label="Select a research report",
            interactive=True
        )

        # Full report text box
        with gr.Group():
            report_txt = gr.Textbox(
                label="Full report text",
                lines=15,
                interactive=False
            )

            # Sources HTML panel
            with gr.Accordion("Sources", open=False):
                sources_html = gr.HTML(
                    label="Sources"
            )

        # Hidden state to remember current index i
        current_idx = gr.Textbox(visible=False)

         # Add visualization button and output
        with gr.Group():
            visualize_btn = gr.Button("Visualize Research Forest", variant="secondary")
            forest_visualization = gr.HTML(label="Research Forest Structure")


        # When question changes: load its reports; clear report_txt & sources
        question_dd.change(
            fn=load_reports_for_question,
            inputs=question_dd,
            outputs=[report_dd, report_txt, sources_html, current_idx],
            queue=False
        )

        # When a report is chosen: show its full text + sources
        report_dd.change(
            fn=display_report_and_sources,
            inputs=report_dd,
            outputs=[report_txt, sources_html],
            queue=False
        )

        # When visualize button clicked: load corresponding forest_i.html
        visualize_btn.click(
            fn=load_forest_html,
            inputs=[current_idx],
            outputs=[forest_visualization],
            queue=False
        )

    return interface