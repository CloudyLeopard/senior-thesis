from textwrap import dedent

from matplotlib.dates import SU


# this prompt is for the librarian, not directly for GPT
BACKGROUND_QUERY_LIBRARIAN = dedent(
    """\
    Generate a background report based on the following research question: {query}. 
    
    Identify the key firms and major events mentioned or implied in the question. Provide background information on each, find any relevant news articles related to them, and include a basic financial analysis for each important firm.
    """
)

DOMAIN_EXPERTS_SYSTEM = dedent(
    """\
    # Goal
    Given a research question, determine distinct domain experts who would answer the question differently, from their unique perspectives.

    # Instructions

    Given a research question, identify domain experts who can each offer valuable, distinct, and non-overlapping perspectives. Each expert should represent a unique field, background, or way of thinking — avoid generic roles or similar types of experts. Each domain expert should have a different answer to the research question. Think creatively and draw from a wide range of disciplines, industries, and worldviews to ensure diversity in insights and approaches.
    
    Approach this iteratively: after identifying one expert, deliberately seek out the next expert from a completely different domain who can contribute a fresh, unique angle to the question. Each expert should add something new — their perspective should not echo or overlap with others.
    
    Your thinking should be thorough and so it's fine if it's very long. You should critically think step by step before and after each domain expert you have decided on.

    You MUST iterate and keep going until you have cannot think of any more unique domain experts. 

    # Output Format
    For each iteration, you should output the following:
    Thoughts 1: [Your thoughts, reasoning through what kind of domain expert would be useful for this question and bring new perspectives, as well as what kind of experts you have already identified]
    Expert Title 1: [The title of your chosen domain expert, like "Automotive Industry Expert"]
    Expert Description 1: [A brief description of the expert's background, focusing on how their expertise relates to the question. Format it like a profile summary, like "The {{Expert Title}} is ..."]

    # Examples

    ## Example 1
    Generate distinct domain experts for the following question:
    How will you value Tesla today?

    Thoughts 1: Tesla is an electric car company, which requires insights from the car industry.
    Expert Title 1: Automotive Industry Expert
    Expert Description 1: The Automotive Industry Expert is a seasoned professional with deep knowledge of the electric vehicle market, including trends, challenges, and innovations in the automotive sector.
    Thoughts 2: Thinking from a different angle, Elon Musk is a controversial political figure, and his political actions alienate consumers from purchasing brands related to him.
    Expert Title 2: Politics Expert
    Expert Description 2: The Politics Expert specializes in the intersection of business and politics, analyzing how political figures influence consumer behavior and brand perception.
    ...

    ## Example 2
    Generate distinct domain experts for the following question:
    How should Netflix evolve its business model in the next 5 years?

    Thoughts 1: Netflix is a media company, and its business model is affected by the media industry.
    Expert Title 1: Media Expert
    Expert Description 1: The Media Expert is a seasoned professional with deep knowledge of the media industry, including trends, challenges, and innovations in content distribution and consumption.
    Thoughts 2: Completely different from the media and business side, Netflix's business model is also affected by consumer behavior, so we need to consider how consumers interact with media platforms.
    Expert Title 2: Cultural Anthropologist
    Expert Description 2: The Cultural Anthropologist studies how cultural trends and consumer behavior shape media consumption, providing insights into evolving audience preferences and engagement strategies.
    Thoughts 3: Netflix's business model is also closely tied to new trends in generative AI. It is important to know how generative media and interactive storytelling may disrupt traditional content pipelines.
    Expert Title 3: AI Narrative Designer
    Expert Description 3: The AI Narrative Designer specializes in the intersection of artificial intelligence and storytelling, exploring how generative AI can create immersive, interactive media experiences that redefine content creation and audience engagement.
    ...

    """
)

DOMAIN_EXPERTS_USER = dedent(
    """\
    Generate distinct domain experts for the following question:
    {query}
    """
)

CHOOSE_EXPERTS_SYSTEM = dedent(
    """\
    # Instructions
    Given a research question, and a list of domain experts, select {n} domain experts whose perspectives would be the most VALUABLE and DISTINCT in answering the question. Make sure no two experts have overlapping experiences and perspectives. The experts should be selected on two criteria: 1) the value of their perspective in answering the question, and 2) the distinctiveness of their perspective compared to other experts.

    # Output Format:
    Thoughts: [Your thoughts, as long as needed, reasoning through which experts would be the most valuable and distinct in answering the question.]
    Selected Experts: [List of selected experts using their original title, separated by commas]
    """
)

CHOOSE_EXPERTS_USER = dedent(
    """\
    Determine the top {n} domain expets from the following list that would provide the most valuable and distinct perspectives on a research question
    
    Research question:
    {query}

    Expert Choices:
    {experts}
    """
)

SUMMARIZE_REPORTS_SYSTEM = "You are a meticulous and insightful research analyst that reviews and summarizes reports."

SUMMARIZE_REPORTS_USER = dedent(
    """\
    Given a research question and a list of research reports that answers the research question, summarize the key findings and insights from each report. You should group similar ideas together, and highlight any conflicting or contradictory information. Your summary should be clear, concise, and easy to understand.

    Research question:
    {query}

    Research reports:
    {reports}

    Identify all the different answers from the reports, and summarize the key findings and insights from each report;
    """
)

# ===== DEFUNCT PROMPTS =====

CREATE_LEAD_SYSTEM = dedent(
    """\
    You are a meticulous and insightful research analyst. Your role is to carefully examine research reports and a main research question to identify unusual, surprising, or noteworthy observations. From these observations, you will develop distinct, actionable, and creative research leads, clearly connected to the original findings. Prioritize cohesive storytelling, insightful reasoning, and avoid generic or superficial statements.
    """
)

CREATE_LEAD_USER = dedent(
    """\
    -Goal-
    The goal is to systematically identify unusual or unexpected observations within preliminary research reports and leverage them into {n} distinct, insightful, and actionable research leads. This approach prioritizes creative reasoning and cohesive storytelling to uncover opportunities and hidden trends. Your analysis should clearly articulate the logical progression from observation -> research lead -> working hypothesis, ensuring each step is precise, actionable, and insightful.

    -Instruction-
    Given the main research question and a preliminary background report, perform the following steps systematically:
    1. Identify unusual, surprising, or notable observations—specifically points or trends that deviate from typical industry patterns, established norms, or expected results. Clearly state why each identified observation stands out as noteworthy or odd.

    2. Develop around {n} concise, actionable research leads from each of these observations. Each lead should
    - Be simple, straightforward, and distinct from one another.
    - Clearly articulate how it emerges from or is connected to the specific observation identified.
    - Avoid generic or overly broad statements; instead, prioritize cohesive storytelling and creative, nuanced lines of reasoning.

    3. For each of the {n} research lead, generate a clear and specific working hypothesis:
	- The hypothesis should logically follow from the research lead.
	- It should be testable or investigable with further analysis or data collection.

    3. Structure your response as follows:
    Observation 1: [Clearly state the unusual or notable observation.]
    Research Lead 1: [Brief, actionable insight or hypothesis directly connected to Observation 1.]
    Working Hypothesis 1: [Clear, concise, testable hypothesis derived directly from Research Lead 1.]

    Observation 2: [Clearly state the unusual or notable observation.]
    Research Lead 2: [Brief, actionable insight or hypothesis directly connected to Observation 2.]
    Working Hypothesis 2: [Clear, concise, testable hypothesis derived directly from Research Lead 2.]

    (Continue as necessary, or until you reached {n} leads)

    -Examples-
    Observation 1: Earnings are weak, signaling vulnerability to a recession, yet the stock price remains surprisingly resilient.
    Research Lead 1: Investigate market expectations about upcoming strategic initiatives or potential improvements that could offset recession concerns.
    Working Hypothesis 1: Investors are anticipating a specific catalyst, such as an innovative product launch or strategic restructuring, which explains the stock's resilience despite current weak earnings.

    Observation 2: Significant earnings growth observed over the past three years.
    Research Lead 2: Identify specific divisions or operational strategies that are driving this growth.
    Working Hypothesis 2: Recent earnings growth has been primarily driven by improvements in operational efficiency or strong market positioning within a particular division, indicating a sustainable competitive advantage.

    Observation 3: Plastics division specifically demonstrates pronounced earnings growth recently.
    Research Lead 3: Explore external factors, such as regulatory changes or shifts in consumer demand, uniquely benefiting the plastics division.
    Working Hypothesis 3: Recent regulatory pressures toward automotive lightweighting have driven increased demand and profitability within the plastics division.

    Observation 4: Earnings improvement in plastics division specifically linked to automobile downsizing.
    Research Lead 4: Investigate long-term trends in automotive manufacturing practices that involve vehicle downsizing or lightweighting.
    Working Hypothesis 4: A sustained automotive industry shift toward smaller, lighter vehicles will continue to boost demand for plastic-based components, ensuring continued growth in the plastics division.

    -Input-
    Research question: 
    {query}

    Research report: 
    {report}

    -Output-
    """
)
