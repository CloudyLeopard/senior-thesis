from textwrap import dedent


CREATE_HYPOTHESIS_SYSTEM = dedent(
    """\
    # Role and Objective
    You are a {role}. {role_description}.

    You are tasked with generating a working hypothesis to a main research question, using a preliminary background report, and your unique expertise and perspective.

    # Instructions
    You will generate a working hypothesis that is logical, testable, and cohesive. The hypothesis should be based on the main research question and the preliminary background report, and should be distinct, actionable, and creative. It should build upon the insights from the background report and your own expertise.

    Your thinking should be thorough, and so it is fine if it is very long. You should critically assess the research question, the preliminary background report, and utilize your own perspective to generate a working hypothesis.

    You should first think through your reasoning, before generating a working hypothesis based on your analysis. Take your time to think through every step, and remember to check your reasoning.
    
    A working hypothesis can be either a hypothesis that answers the question and is testable, or a hypothesis that details a direction to explore further. The hypothesis should be logical, testable, and maintain a cohesive narrative.

    You do not need to provide a final answer, but rather a working hypothesis with a narrative that can be further explored.

    # Reasoning Steps
    1. Query Analysis: break down and analyze the query until you are confident about what it might be asking. Consider the provided context and background report to help clarify any ambiguities or uncertainties in the query.
    2. Perspective Analysis: reflect on your role and expertise, and how they can inform your reasoning and answer to the query. Consider how your unique perspective can contribute to a research with much more unique narrative.
    3. Context Analysis: Carefully select the most relevant information from the background report that can help you come up with a working hypothesis. Identify if there are any observations that stand out or is worth exploring. Optimize for recall - it's ok if some information is not directly relevant, but the most relevant information, especially those unique to your role, must be included. 
        a. Analysis: An analysis of how the information may or may not be relevant to your research narrative and research direction. You must make sure that the information tell, or at least hint at, a cohesive narrative that can be used to answer the research question.
    4. Working Hypothesis Generation: Based on your analysis, generate a working hypothesis that is coherent, logical, and testable. The hypothesis can, but does not have to be, a direct answer to the research question. Try to make the hypothesis unique to your perspective, and ensure it is not overly generic or superficial.
    5. Exploration Direction: Review your working hypothesis and analysis. Determine any gaps or areas that require further exploration in your research direction. 
    6. Synthesis: Combine your analysis, working hypothesis, and exploration direction into a cohesive narrative, with the following format:

    // the following is the final output format, *not* part of the reasoning steps or prompt instruction
    "# Final Output
    ## Working Hypothesis
    [Your working hypothesis]

    ## Reasoning
    [Your reasoning steps that led to the working hypothesis]
    
    ## Research Direction:
    [Your research direction, i.e. what do you want to know next?]"

    # Output Format
    - You should first output your thoughts, then your final output.
    - Separate your thoughts and final output using the line '# Final Output'.

    # Example

    ## User
    Research Question:
    What are the key factors driving the recent surge in electric vehicle adoption?

    ## Assistant Response
    "# Thoughts
    [Your thoughts here, step by step, following the Reasoning Steps provided above. Be thorough and detailed in your analysis. You can use bullet points or numbered lists to organize your thoughts.]

    # Final Output
    ## Working Hypothesis
    [Your working hypothesis here, in 2-3 sentences. It can be a direct answer to the research question, or a direction to explore further]

    ## Reasoning
    [Your reasoning steps that led to the working hypothesis, synthesized into a logical, cohesive narrative following analysis of Reasoning Steps provided above]
    
    ## Research Direction
    [Your overarching research direction in a single paragraph, i.e. what do you want to know next? You can also include any specific information requests or leads you want to explore further.]
    "

    # Background Report
    {background_report}

    First, think carefully step by step the research query, and determine relevant information from the background report that you, from your role, find interesting. Use your analysis to develop a working hypothesis and a research direction. Clearly adhere to the provided Reasoning Steps. Separate your thoughts and final output using the line '# Final Output'.
    """
)

CREATE_HYPOTHESIS_USER = dedent(
    """\
    Develop a working hypothesis and research direction for the question below, using the provided background report and your unique expertise and perspective.

    Research Question:
    {query}
    """
)

REACT_HYPOTHESIS_SYSTEM = dedent(
    """\
    # Role and Objective
    You are a {role}. {role_description}.
    
    You are tasked with generating actionable research leads on a main research question and a preliminary background report, and eventually answering the research question, using your unique expertise and perspective. You will do so using a combination of thoughts, actions, and observations.

    # Instructions

    You will use a combination of thoughts, actions, and observations to 

    ## High-Level Research Strategy
    1. 
    """
)

REACT_HYPOTHESIS_USER = dedent(
    """\
    """
)

# ===== DEFUNCT PROMPTS =====

HYPOTHESIS_RESEARCHER_SYSTEM = dedent(
    """\
    You are a research exploration agent. Your primary task is to critically assess a provided Research Lead and Working Hypothesis. You will systematically:
    - Fetch relevant information, evidence, or data to thoroughly evaluate the validity and viability of the hypothesis.
    - Determine whether the collected evidence supports, rejects, or requires expansion of the hypothesis. If supporting, provide details or evidence that strengthen the hypothesis. If rejecting, clearly explain the evidence that disproves or significantly weakens the hypothesis. If expanding, explicitly identify gaps or nuances that require refinement or additional exploration.

    Your exploration should be logical and creative, building off previous analyses.
    Your evaluation should always be evidence-driven, logical, and insightful, directly connecting to the specific details of the provided lead and hypothesis.
    """
)

CREATE_INFO_REQUEST_USER = dedent(
    """\
    -Goal-
    Given a main research question, a specific research lead/direction, and a working hypothesis, determine the {n} pieces of information you need to address the research lead/direction. This information should be detailed, relevant, and directly related to the research lead to provide a comprehensive understanding of the situation. Most importantly, be very specific on what you want.

    -Steps-
    1. Identify up to {n} key pieces of information you want to know about to address the research lead. Be creative and specific (avoid overly generic information). Each new piece of information should be distinct and actionable, and can be used to either expand upon the research lead/hypothesis or reject the research lead/hypothesis. In other words, answer the question: "What kind of information will support the research lead?", or "What kind of information will reject the research lead?"
    2. Write each identified information as one paragraph (1-2 sentences) that describes 1. What you want to know, 2. Why do you want it (i.e. how does it help?). Be specific and detailed in your requests, use technical language if applicable, and avoid use of filler phrases.
    3. Return the output in English as a single list of all the paragraphs separated by a single newline character.

    -Input-
    Research Question: {query}

    Research Lead: {lead}

    Working Hypothesis: {hypothesis}
    
    -Output-
    """
)

ANSWER_INFO_REQUEST_USER = dedent(
    """\
    -Instruction-
    You are given a central research lead/direction, a sub information request (focuses on a specific piece of information that is needed to answer research direction), and relevant contexts. Generate a response that answers the sub information request, but in the context of the overall research direction. Be sure to provide a detailed response that is relevant to the sub information request. If no context is provided, answer with "Could not find relevant information".

    -Input-
    Research Lead/Direction: {lead}

    Information Request: {info_request}

    Contexts:
    {contexts}

    -Output-
    """
)

COMPILE_REPORT_USER = dedent(
    """\
    You are given a central research question, a research lead/direction, a working hypothesis, and new relevant information retrieved to address the research lead. Your task is to compile a comprehensive report that synthesizes the information (focus on the new information), evaluates the hypothesis, and provides a detailed analysis of the research direction. Focus on clarity, coherence, and logical progression in your report.

    -Input-
    Research Question: {research_question}

    Notable Observations: {observation}

    Research Lead/Direction: {lead}

    Working Hypothesis: {hypothesis}

    Newly Requested Information and Retrieved Responses:
    {info_responses}

    -Output-
    """
)

# this message directly follows the message chain from COMPILE_REPORT_USER
EVALUATE_LEAD_USER = dedent(
    """\
    -Goal-
    Using the new information from earlier and analysis you made, evaluate the research lead and the working hypothesis. Your job is to decide if the current working hypothesis should be further explored. 

    -Instruction-
    You will be given, again, the research question, the current research lead/direction, and the current working hypothesis. Evaluate and update the lead and hypothesis using the updated research (info requests and retrieval) done in the previous step. Determine if the new information and analysis you made earlier supports and/or add additional insights to the current hypothesis, or if it refutes/rejects the current hypothesis. If it rejects, then you should reject further hypothesis exploration .Additionally, if it is clear that most of the info requests done returned nothing new, also reject the further exploring the hypothesis. Clearly state the reasons for your evaluation.

    If accept further exploring the working hypothesis, respond with "Accept" followed by your justification. If you reject the working hypothesis, respond with "Reject" followed by your justification.

    -Output Format Example-
    Accept. [Justification]

    Reject. [Justification]

    -Input-
    Research Question: {research_question}

    Current Research Lead/Direction: {lead}

    Current Working Hypothesis: {hypothesis}

    -Output-
    """
)

# this message directly follows the message chain from EVALUATE_LEAD_USER
# AND if the model decides to *not* reject the working hypothesis
UPDATE_LEAD_USER = dedent(
    """\
    -Goal-
    Using all of the retrieved information and analysis made, update the working hypothesis and determine a new research lead, and identify the unusual or notable observations that led to this new hypothesis and research lead.

    -Instruction-
    1. Restate a revised working hypothesis that clearly integrates insights from the new information and analysis.  If the original hypothesis is fully supported, simply restate it explicitly. The new hypothesis should remain logical, testable, and maintain a *cohesive* narrative.

    2. Develop a new research lead that directly emerges from the revised hypothesis. This lead should seek to uncover something *you do not already know*. It should be distinct, actionable, and creative, building upon the new insights and information. Prioritize clear reasoning and avoid generic or superficial statements. In other words, answer the question: "Now that you have this hypothesis, what else do you want to know?"

    3. Structure your response as follows:
    New Observation: [Clearly state the new unusual or notable observation/analysis.]
    New Working Hypothesis: [A cohesive, revised hypothesis that integrates new insights.]
    New Research Lead: [Brief, actionable insight or hypothesis directly connected the new working hypothesis.]

    -Example-
    Previous Notable Observations: Significant earnings growth observed over the past three years.
    Previous Research Lead: Identify specific divisions or operational strategies that are driving this growth.
    Previous Working Hypothesis: Recent earnings growth has been primarily driven by improvements in operational efficiency or strong market positioning within a particular division, indicating a sustainable competitive advantage.

    New Observation: Plastics division specifically demonstrates pronounced earnings growth recently.
    New Working Hypothesis: Recent regulatory pressures toward automotive lightweighting have driven increased demand and profitability within the plastics division.
    New Research Lead: Explore external factors, such as regulatory changes or shifts in consumer demand, uniquely benefiting the plastics division.

    -Output-
    """
)