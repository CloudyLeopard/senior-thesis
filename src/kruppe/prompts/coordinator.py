from textwrap import dedent

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
