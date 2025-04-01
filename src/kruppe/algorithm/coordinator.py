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
)
from kruppe.utils import log_io
from kruppe.models import Response

logger = logging.getLogger(__name__)

class Coordinator(Researcher):
    librarian: Librarian
    research_question: str
    background_report: str | Response = None
    bkg_researcher: BackgroundResearcher = None
    num_leads: int = 3
    hyp_researcher_config: Dict = {}
    _research_queue: List[Lead] = PrivateAttr([])
    _hyp_researchers: List[HypothesisResearcher] = PrivateAttr([])

    @model_validator(mode='after')
    def check_background_report(self):
        if self.background_report is None and self.bkg_researcher is None:
            raise ValueError("Either a background report or a background researcher must be provided.")
        
        if self.bkg_researcher is not None:
            self.background_report = self.bkg_researcher.latest_report # could be None still, if bkg_researcher did not run anything yet
        return self

    async def execute(self) -> List[Dict[str, Any]]:

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
