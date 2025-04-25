import asyncio
import re
from pydantic import BaseModel, computed_field, PrivateAttr, model_validator
from typing import Callable, List, Dict, Any
import logging
from tqdm import tqdm

from kruppe.llm import OpenAILLM
from kruppe.algorithm.agents import Researcher
from kruppe.algorithm.hypothesis import HypothesisResearcher
from kruppe.algorithm.librarian import Librarian
from kruppe.prompts.coordinator import (
    BACKGROUND_QUERY_LIBRARIAN,
    DOMAIN_EXPERTS_SYSTEM,
    DOMAIN_EXPERTS_USER,
    CHOOSE_EXPERTS_SYSTEM,
    CHOOSE_EXPERTS_USER,
    SUMMARIZE_REPORTS_SYSTEM,
    SUMMARIZE_REPORTS_USER
)
from kruppe.common.log import log_io
from kruppe.models import Response

logger = logging.getLogger(__name__)

class Coordinator(Researcher):
    librarian: Librarian
    tree_configs: Dict[str, Any] = {
        "llm": OpenAILLM(),
        "toolkit": [],
        "max_step": 10,
        "max_degree": 3,
        "docstore": None,
        "index": None
    }

    research_reports: List[Response] = []
    
    _background_report: Response = PrivateAttr(default=None)
    _research_forest: List[HypothesisResearcher] = PrivateAttr(default=None)

    def reports_to_dict(self) -> Dict[str, Any]:
        """Convert research reports to dictionaries.
        """

        research_report_dicts = [
            report.model_dump(mode='json') for report in self.research_reports
        ]

        return {
            "research_reports": research_report_dicts,
            "background_report": self._background_report.model_dump(mode='json') if self._background_report else None,
        }



    async def generate_background(self, query: str) -> Response:
        """Generate a background report using the librarian.

        Returns:
            Response: Librarian's response containing the background report.
        """

        if self._background_report:
            logger.debug("Using cached background report.")
            return self._background_report
        
        logger.debug("Generating background report using librarian.")

        prompt = BACKGROUND_QUERY_LIBRARIAN.format(query=query)
        librarian_response = await self.librarian.execute(prompt)


        return librarian_response
    
    async def generate_domain_experts(self, query: str) -> Dict[str, str]:
        """Generate a list of domain experts based on the research question.

        Args:
            query (str): The research question.

        Returns:
            Dict[str, str]: A dictionary mapping expert names to their expertise.
        """

        # GENERATE SAMPLE DOMAIN EXPERTS

        messages = [
            {"role": "system", "content": DOMAIN_EXPERTS_SYSTEM},
            {"role": "user", "content": DOMAIN_EXPERTS_USER.format(query=query)}
        ]

        domains_response = await self.llm.async_generate(messages)
        domains_str = domains_response.text

        
        # PARSE DOMAIN EXPERTS
        experts = {}

        curr_title = None
        curr_desc = None
        for line in domains_str.splitlines():
            if not line or ':' not in line:
                continue
            
            # Fast prefix checks
            if line.startswith("Expert Title"):
                title = line.split(':', 1)[1].strip()

                if curr_title is None:
                    curr_title = title
            
            elif line.startswith("Expert Description"):
                desc = line.split(':', 1)[1].strip()

                if curr_desc is None:
                    curr_desc = desc
            
            if curr_title and curr_desc:
                experts[curr_title] = curr_desc
                curr_title = None
                curr_desc = None
        return experts
    
    async def filter_domain_experts(self, query: str, experts: Dict[str, str], n_experts: int = 5) -> Dict[str, str]:

        if len(experts) == 0:
            logger.warning("No domain experts found.")
            return {}


        if len(experts) <= n_experts:
            logger.info("No need to filter domain experts.")
            return experts


        # SELECT THE BEST DOMAIN EXPERTS
        expert_choices = "\n".join(f"{title}: {desc}" for title, desc in experts.items())
        messages = [
            {"role": "system", "content": CHOOSE_EXPERTS_SYSTEM},
            {"role": "user", "content": CHOOSE_EXPERTS_USER.format(
                query=query,
                n=n_experts,
                experts=expert_choices
            )}
        ]
        choose_response = await self.llm.async_generate(messages)
        choose_str = choose_response.text

        # Parse the chosen experts
        experts_str = choose_str.strip().split("Selected Experts:")[1].strip()
        selected_experts_titles = [title.strip() for title in experts_str.split(',') if title.strip()]
        selected_experts = {title: experts[title] for title in selected_experts_titles if title in experts}

        logger.debug(f"Selected {len(selected_experts)} domain experts for query '{query}'")
        return selected_experts
        
    async def initialize_research_forest(
        self, 
        query: str,
        background: Response,
        n_experts: int = 5,
    ) -> List[HypothesisResearcher]:

        # Generate domain experts
        experts = await self.generate_domain_experts(query)
        filtered_experts = await self.filter_domain_experts(query, experts, n_experts)
        print(f"Domain experts generated: {len(filtered_experts)} experts found from {len(experts)} generated.")

        # Initialize hypothesis researchers or "research forest"
        hyp_researchers = []
        for expert_name, expert_desc in filtered_experts.items():
            hyp_researcher = HypothesisResearcher(
                role=expert_name,
                role_description=expert_desc,
                background_report=background,
                **self.tree_configs,
            )
            hyp_researchers.append(hyp_researcher)
    
        
        return hyp_researchers

    

    async def execute(self, query: str, n_experts: int = 5) -> List[Response]:
        """Execute the coordinator's research process.

        Args:
            query (str): The research question.

        Returns:
            Response: The final response containing the research results.
        """
        # reset research reports
        self.research_reports = []

        # Generate background report
        # not resetting background report in case user manually sets it
        self._background_report = await self.generate_background(query)
        print("Background report generated for query:", query)
        
        # Initialize hypothesis researchers
        hyp_researchers = await self.initialize_research_forest(query, self._background_report, n_experts)
        self._research_forest = hyp_researchers

        # Execute hypothesis researchers in parallel
        async def execute_hypothesis_researcher(hyp_researcher: HypothesisResearcher, query: str):
            responses = await hyp_researcher.execute(query=query)
            self.research_reports.extend(responses)
            print(f"Researcher {hyp_researcher.uuid} completed with {len(responses)} reports.")
        
        async with asyncio.TaskGroup() as tg:
            for hyp_researcher in hyp_researchers:
                tg.create_task(execute_hypothesis_researcher(hyp_researcher, query))
        
        print(f"Completed {len(self.research_reports)} research reports for query: {query}.")

        # # Execute hypothesis researchers sequentially
        # for hyp_researcher in hyp_researchers:
        #     reports = await hyp_researcher.execute(query=query)

        #     self.research_reports.extend(reports)
        #     print(f"Researcher {hyp_researcher.role} completed with {len(reports)} reports.")

        
        return self.research_reports

    async def summarize_reports(self) -> Response:
        """Summarize the research reports.

        Returns:
            Response: The summary of the research reports.
        """

        if len(self.research_reports) == 0:
            logger.warning("No research reports to summarize.")
            return None
        
        # Summarize the research reports

        assert len(set(report.metadata['query'] for report in self.research_reports)) == 1, "All reports must have the same query."

        query = self.research_reports[0].metadata['query']
        reports_str = "\n----------\n".join(report.text for report in self.research_reports)

        messages = [
            {"role": "system", "content": SUMMARIZE_REPORTS_SYSTEM},
            {"role": "user", "content": SUMMARIZE_REPORTS_USER.format(query=query, reports=reports_str)}
        ]

        summary_response = await self.llm.async_generate(messages)

        return summary_response

    def visualize_research_forest(self):
        """
        Generates an HTML visualization of the research forest structure.
        Each HypothesisResearcher is a root, with its root nodes and their children forming a tree.
        """

        if not self._research_forest:
            return "<div>No research forest available yet.</div>"
        
        html = """
        <style>
            .tree {
                font-family: Arial, sans-serif;
                margin: 20px;
                color: black;
            }
            .tree ul {
                list-style-type: none;
                padding-left: 20px;
                position: relative;
            }
            .tree ul:before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                border-left: 1px solid #ccc;
                height: 100%;
            }
            .tree li {
                position: relative;
                padding: 5px 0;
            }
            .tree li:before {
                content: "";
                position: absolute;
                top: 50%;
                left: -20px;
                border-top: 1px solid #ccc;
                width: 20px;
            }
            .node {
                padding: 5px 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f8f9fa;
                display: inline-block;
            }
            .researcher {
                background: #e3f2fd;
                font-weight: bold;
            }
            .root-node {
                background: #f3e5f5;
            }
            .child-node {
                background: #f1f8e9;
            }
            .leaf_node {
                background: #fff3e0;
            }
        </style>
        <div class="tree">
        """
        
        for i, researcher in enumerate(self._research_forest):
            html += f"""
            <ul>
                <li>
                    <div class="node researcher">Hypothesis Researcher {i+1}</div>
                    <ul>
            """
            
            # Add root nodes
            for root_node in researcher.root_nodes:
                html += f"""
                        <li>
                            <div class="node root-node">Root Node: step={root_node.step}, d_time={root_node.d_time}</div>
                            <ul>
                """
                
                # Add children recursively
                def add_children(node):
                    html = ""
                    for child in node.children:
                        if child.is_leaf:
                            html += f"""
                                <li>
                                    <div class="node leaf_node">Leaf Node: step={child.step}, d_time={child.d_time}</div>
                            """
                        else:
                            html += f"""
                                    <li>
                                        <div class="node child-node">Internal Node: step={child.step}, d_time={child.d_time}</div>
                            """
                        if child.children:
                            html += "<ul>"
                            html += add_children(child)
                            html += "</ul>"
                        html += "</li>"
                    return html
                
                html += add_children(root_node)
                html += """
                            </ul>
                        </li>
                """
            
            html += """
                    </ul>
                </li>
            </ul>
            """
        
        html += "</div>"
        return html 
