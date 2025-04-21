import re
from pydantic import BaseModel, computed_field, PrivateAttr, model_validator
from typing import Callable, List, Dict, Any
import logging
from tqdm import tqdm

from kruppe.algorithm.agents import Researcher
from kruppe.algorithm.hypothesis import HypothesisResearcher
from kruppe.algorithm.librarian import Librarian
from kruppe.prompts.coordinator import (
    BACKGROUND_QUERY_LIBRARIAN,
    DOMAIN_EXPERTS_SYSTEM,
    DOMAIN_EXPERTS_USER,
    CHOOSE_EXPERTS_SYSTEM,
    CHOOSE_EXPERTS_USER
)
from kruppe.common.log import log_io
from kruppe.models import Response

logger = logging.getLogger(__name__)

class Coordinator(Researcher):
    librarian: Librarian
    _background_report: Response = PrivateAttr(default=None)


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
            logger.debug("No need to filter domain experts.")
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
        
    def initialize_research_forest(
        self, 
        experts: Dict[str, str],
        toolkit: List[Callable]
    ) -> List[HypothesisResearcher]:
        # TODO: generate initial lead or hypothesis based on expert and background report
        
        hyp_researchers = []
        for expert_name, expert_desc in experts.items():
            hyp_researcher = HypothesisResearcher(
                llm=self.llm,
                toolkit=toolkit,
                role=expert_name,
                role_description=expert_desc,
            )
            hyp_researchers.append(hyp_researcher)
        
        return hyp_researchers

    async def execute(self, query: str, n_experts: int = 5) -> Response:
        """Execute the coordinator's research process.

        Args:
            query (str): The research question.

        Returns:
            Response: The final response containing the research results.
        """
        
        # Generate background report
        background_report = await self.generate_background(query)
        self._background_report = background_report
        logger.debug("Background report generated.")

        # Generate domain experts
        experts = await self.generate_domain_experts(query)
        filtered_experts = await self.filter_domain_experts(query, experts, n_experts)
        logger.debug(f"Domain experts generated: {len(filtered_experts)} experts found from {len(experts)} generated.")

        # Initialize hypothesis researchers
        toolkit = self.librarian.toolkit # using librarian's toolkit for hypothesis researchers
        hyp_researchers = self.initialize_research_forest(filtered_experts, toolkit)

        # Execute hypothesis researchers
        # TODO: parallelize this, but max like 2 at a time.
        for hyp_researcher in hyp_researchers:
            await hyp_researcher.execute(
                query=query,
                background=background_report.text,
            )

        # TODO: group reports by the content of the report/narrative
        final_report = Response(text="Research completed successfully.")
        
        return final_report