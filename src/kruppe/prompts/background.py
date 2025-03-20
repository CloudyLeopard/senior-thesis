from textwrap import dedent


RESEARCH_STANDARD_SYSTEM = dedent(
    """\
    You are an expert at business and finance research. Your primary objective is to determine how to thoroughly gather the necessary research to answer a question.
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

        Format each entity as ("entity"|<entity_name>|<entity_category>|<reasoning>)

        3. For each entity identified in step 1, infer the background information about that entitiy that a person needs additional research on to be able to fully and accurately anser the query. Be creative and specific (avoid overly general information). Each piece of information should be absolutely necessary and adds to the overall background research process.

        4. For each background information from step 3, extract the following information:
        - description: the description of the information that requires additional research
        - information_category: the category of the piece of information. Should be one of the following types: {information_categories} or others (provide your own category)
        - reasoning: justification for choosing to research this area.
        Format each information as ("info"|<description>|<information_category>|<reasoning>|<entity_name>)

        5. Return the output in English as a single list of all identified entities (from step 1) and identified information (from step 3), separated by a newline character.

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
        Format each information as ("info"|<description>|<information_category>|<reasoning>)

        3. Return the output in English as a single list of all identified information separated by a newline character.

        -Input-
        Query: {query}
    
        -Output-
        """
    )
]

ASSIGN_TOOL_USER = dedent(
    """\
    -Goal-
    Given a request for information and a list of tools, determine which tools an AI agent should use to conduct research for the request for information. 

    -Steps-
    1. For each tool, use your intuition and the tool description to infer the type of information that the tool can provide for the request.
    For each tool, determine the following information:
    - tool_name: the tool name in question. use the original name provided in the input
    - research_area: the area of research that this tool can assist with to learn more about the request, and its importance.
    - rank: the importance of using this tool, where 1 is very important and must be used, and 3 is least important and can be ignored. Minimize the number of tools assigned with a rank of 1, unless absolutely necessary.

    Format each tool identification as ("tool"|<tool_name>|<research_area>|<rank>)

    2. Return output in English as a single list of all the tools identified in step 1, **ordered from most important to least important**, separated by a newline character.

    -Input-
    Request for Information:
    {info_request}

    Tools Descriptions:
    {tools_descriptions}

    -Output-
    """
)