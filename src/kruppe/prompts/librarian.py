from textwrap import dedent

LIBRARIAN_STANDARD_SYSTEM = dedent(
    """\
    You are an expert at business and finance research. You will be given a list of resources/functions, and your task is to determine which function needs to be used to achieve a certain objective, and the parameters needed to execute the function.
    """
)

LIBRARIAN_STANDARD_USER = dedent(
    """\
    -Goal-
    Given a description of the information that the user is seeking, the list of resources/functions that can be used to gather the information, and a history of past resources/functions call that were already made, determine the functions that should be used to find the information and the parameters needed to execute the function. You should *not* repeat function calls with the same or very similar parameters.

    -Steps-
    1. For each resource/function, use your intuition and the function description to infer the type of information that the function can provide for the request, and determine the parameters needed to execute the function. Make sure that you are not making funtion calls that have already been made in the past.
    For each resource, determine the following information:
    - func_name: the function name in question. use the original name provided in the input
    - parameters: the parameters that will be entered to the function. It should be a stringified object.
    - purpose: describe what you want to learn by using this resource/function
    - rank: the importance of using this resource, where 1 is very important and must be used, and 3 is least important and can be ignored. Minimize the number of functions assigned with a rank of 1, unless absolutely necessary.

    Format each resource identification as (resc|<func_name>|<parameters>|<purpose>|<rank>|).

    2. Return output in English as a single list of all the resources identified in step 1, **ordered from most important to least important**, separated by a newline character.

    -Input-
    Description of the information that the user wants to know:
    {information_desc}

    Resource/Function Descriptions in JSON Format:
    {resource_desc}

    Past Resource/Function Calls:
    {funcs_past}

    -Output-
    """
)

LIBRARIAN_TIME_USER = dedent(
    """\
    -Instruction-
    You are given a query, or essentially a description of the information that the user is seeking. This query will be used to query a vector storage system to retrieve relevant contexts. I want you to analyze the query, and determine the time period that the user is interested in. The time period should be as specific as possible, and should be in the format of a start date and an end date.

    Output the format like follow:
    start_date: <start_date>
    end_date: <end_date>

    -Input-
    Description of the information that the user wants to know:
    {information_desc}

    -Output-
    """
)

# NOTE: this prompt is not used, but later move it to Index
LIBRARIAN_QUERY_USER = dedent(
    """\
    -Goal-
    Given a description of the information that the user is seeking, create {n} queries that is used to query a vector storage system to retrieve relevant documents. The combined results of the {n} queries should answer all aspects of the information request description. Make sure that the {n} queries are mutually exclusive and collectively exhaustive.

    -Steps-
    1. For each query, use your intuition and the information description to infer the type of information that the query should retrieve. Make sure that the queries are mutually exclusive and collectively exhaustive.

    2. Return output in English as a single list of all the queries identified in step 1, separated by a newline character.

    -Input-
    Description of the information that the user wants to know:
    {information_desc}

    -Output-
    """
)