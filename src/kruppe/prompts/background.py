from textwrap import dedent


RESEARCH_STANDARD_SYSTEM = dedent(
    """\
    You are an expert at business and finance research. Your primary objective is to determine how to thoroughly gather the necessary research to answer a question.
    """
)

DETERMINE_INFO_REQUEST_USER = dedent(
    """\
    -Goal-
    Given a query, determine what additional background information you want to know to comprehensively and accurately answer the query.

    -Steps-
    1. Identify all the information you want to know about to answer the query. Be creative and specific (avoid overly general information). Each new piece of information should help build a more complete background for the user to answer the question.
    2. Write each identified information as one paragraph (2-3 sentences) that describes 1. What you want to know, 2. Why do you want it (i.e. how does it help?). Note that your justification does not have only be "to help answer the question". Other ideas include "to gain a better background", "to double check the event's validity", etc.
    3. Return the output in English as a single list of all the paragraphs separated by a single newline character.

    -Input-
    Query: {query}
    
    -Output-
    """
)

ANSWER_INFO_REQUEST_USER = dedent(
    """\
    -Instruction-
    You are given a central research question, a sub informatino request (focuses on a specific piece of information that is needed to answer the central research question), and relevant contexts. Generate a response that answers the sub information request, but in the context of the overall research question. Be sure to provide a detailed response that is relevant to the sub information request.

    -Input-
    Research Question: {query}

    Information Request: {info_request}

    Contexts:
    {contexts}

    -Output-
    """
)

COMPILE_REPORT_USER = dedent(
    """\
    Given a central research question, a list of sub information requests used to answer the research question's background, and answers to these information requests, build a background report of the research question. Do not focus on answering the query. Focus on writing relevant background information that will be later used to help answer the query. Cite specific examples to back up your analysis whenever possible.

    -Input-
    Research Question: {query}

    Information Requests and Responses:
    {info_responses}

    -Output-
    """
)



# ========================

ANALYZE_DOCUMENT_SIMPLE_USER = dedent(
    """\
    -Instruction-
    Given a document, identify and summarize all elements discussed in the topic that could be relevant to the query, even if remotely. Return a concise paragraph of your analysis, which should include 1. a description of the relevant observations (be specific) and 2. an analysis of why these observations could be related to the query. Limit your analysis to 3-4 sentences.

    -Input-
    Query: {query}
    
    <document>
    {document}
    </document>

    -Output-
    """
)

ANALYZE_QUERY_USER_CHAIN = [
    dedent(
        """\
        -Goal-
        Given a query, identify all entities in the query that need additional background research to answer the query to its fullest. 
        
        -Steps-
        1. Identify all entities in the query.
        2. For each identified entity, extract the following information:
        - entity_name: Name of the entity
        - entity_category: the category of the extracted entity. Should be one of the following types: {entity_categories} or others (provide your own category)
        - reasoning: your reasoning to extract this entity, or the entitiy's position in the query. For example it could be the main subject, the main cause, the big unknown, or others.

        Format each entity as (entity|<entity_name>|<entity_category>|<reasoning>)

        3. For each entity identified in step 1, infer the background information about that entitiy that a person needs additional research on to be able to fully and accurately anser the query. Be creative and specific (avoid overly general information). Each piece of information should be absolutely necessary and adds to the overall background research process.

        4. For each background information from step 3, extract the following information:
        - information_description: the description of the information that requires additional research
        - information_category: the category of the piece of information. Should be one of the following types: {information_categories} or others (provide your own category)
        - reasoning: justification for choosing to research this area.
        Format each information as (info|<information_description>|<information_category>|<reasoning>|<entity_name>)

        5. Return the output in English as a single list of all identified entities (from step 1) and identified information (from step 3), separated by a single newline character.

        -Input-
        Query: {query}
        
        -Output-
        """
    ),
    dedent(
        """\
        -Goal-
        Given a query, infer what additional information a person needs to know to be able to fully and accurately answer the query. Be creative and specific (avoid overly general information). Each piece of information should be absolutely necessary and adds to the overall background research process.
        
        -Steps-
        1. Identify all information that requires additional research.
        2. For each identified information, extract the following:
        - description: the description of the information that requires additional research
        - information_category: the category of the piece of information. Should be one of the following types: {information_categories} or others (provide your own category)
        - reasoning: justification for choosing to research this area.
        Format each information as (info|<description>|<information_category>|<reasoning>)

        3. Return the output in English as a single list of all identified information separated by a newline character.

        -Input-
        Query: {query}
    
        -Output-
        """
    )
]