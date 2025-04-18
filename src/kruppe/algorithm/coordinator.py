import re
from pydantic import BaseModel, computed_field, PrivateAttr, model_validator
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from kruppe.algorithm.agents import Researcher, Lead
from kruppe.algorithm.hypothesis import HypothesisResearcher
from kruppe.algorithm.background import BackgroundResearcher
from kruppe.algorithm.librarian import Librarian
from kruppe.prompts.coordinator import (
    CREATE_LEAD_SYSTEM,
    CREATE_LEAD_USER,
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

        self._background_report = librarian_response

        return librarian_response
    
    async def generate_domain_experts(self, query: str, n_experts: int = 5) -> Dict[str, str]:
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

        # SELECT THE BEST DOMAIN EXPERTS
        messages = [
            {"role": "system", "content": CHOOSE_EXPERTS_SYSTEM},
            {"role": "user", "content": CHOOSE_EXPERTS_USER.format(query=query, n=n_experts)}
        ]
        choose_response = await self.llm.async_generate(messages)
        choose_str = choose_response.text

        # Parse the chosen experts
        experts_str = choose_str.strip().split("Selected Experts:")[1].strip()
        selected_experts_titles = [title.strip() for title in experts_str.split(',') if title.strip()]
        selected_experts = {title: experts[title] for title in selected_experts_titles if title in experts}

        logger.debug(f"Selected {len(selected_experts)} domain experts for query '{query}'")
        return selected_experts
        
    async def initialize_research_forest(self, query: str, experts: Dict[str, str]):
        # TODO: generate initial lead or hypothesis based on expert and background report
        ...
        

        

    async def execute(self, query: str) -> Response:





    @model_validator(mode='after')
    def check_background_report(self):
        if self.background_report is None and self.bkg_researcher is None:
            raise ValueError("Either a background report or a background researcher must be provided.")
        
        if self.bkg_researcher is not None:
            self.background_report = self.bkg_researcher.latest_report # could be None still, if bkg_researcher did not run anything yet
        return self

    async def execute_old(self) -> List[Dict[str, Any]]:

        if self.background_report is None:
            print("Executing background researcher...")
            bkg_report = await self.bkg_researcher.execute()
            self.background_report = bkg_report
        
        print("Creating leads...")

        # Create leads
        leads = await self.create_leads(self.research_question, self.background_report)
        self._research_queue.extend(leads)

        print(f"Created {len(leads)} leads")

        # Create hypothesis researchers
        print("Creating hypothesis researchers...")

        for lead in tqdm(self._research_queue, desc="hypothesis researchers"):
            hyp_researcher = HypothesisResearcher(
                new_lead=lead,
                research_question=self.research_question,
                **self.hyp_researcher_config
            )
            self._hyp_researchers.append(hyp_researcher)

        print("Executing hypothesis researchers...")
        
        # Execute hypothesis researchers
        i = 1
        for hyp_researcher in self._hyp_researchers:
            await hyp_researcher.execute()
            print("Finished executing hypothesis researcher", i)
            i += 1

        # return final reports
        print("Compiling reports...")
        results = []
        for i in range(len(self._hyp_researchers)):
            hyp_researcher = self._hyp_researchers[i]
            if hyp_researcher.report_ready:
                results.append({
                    "original_lead": self._research_queue[i],
                    "report": hyp_researcher.latest_report,
                    "hypothesis": hyp_researcher.latest_hypothesis
                })
        
        return results

    @log_io
    async def create_leads(self) -> List[Lead]:
        """
        Given a research question and a preliminary background report, generate research leads.

        Args:
            query (str): research question
            report (str | Response): preliminary background report

        Returns:
            List[Lead]: list of research leads
        """

        if self.background_report is None:
            raise ValueError("A background report must be provided to generate leads.")

        # Convert Response object to text
        report_text = self.background_report
        if isinstance(report_text, Response):
            report_text = report_text.text
        
        user_message = CREATE_LEAD_USER.format(
            query=self.research_question,
            report=report_text,
            n=self.num_leads
        )

        messages = [
            {"role": "system", "content": CREATE_LEAD_SYSTEM},
            {"role": "user", "content": user_message},
        ]

        llm_response = await self.llm.async_generate(messages)
        llm_string = llm_response.text

        # -- REGEX TO PARSE LEADS --
        # This regex pattern captures three groups corresponding to observation, lead, and working hypothesis.
        # It looks for the pattern "Observation <number>:", "Research Lead <number>:", and "Working Hypothesis <number>:".
        pattern = (
            r"\*{0,2}Observation \d+:\*{0,2}\s*(.*?)\s*[\r\n]+"
            r"\*{0,2}Research Lead \d+:\*{0,2}\s*(.*?)\s*[\r\n]+"
            r"\*{0,2}Working Hypothesis \d+:\*{0,2}\s*(.*?)(?:\s*[\r\n]+|$)"
        )
        
        # re.DOTALL ensures that the dot (.) matches newlines as well.
        matches = re.findall(pattern, llm_string, re.DOTALL)
        leads = []
        
        for obs, lead, hypothesis in matches:
            leads.append(Lead(
                observation=obs.strip(),
                lead=lead.strip(),
                hypothesis=hypothesis.strip()
            ))

        return leads
