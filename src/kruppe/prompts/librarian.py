from textwrap import dedent

LIBRARIAN_STANDARD_SYSTEM = dedent(
    """\
    You are an expert at business and finance research. You will be given a list of resources/functions, and your task is to determine which function needs to be used to achieve a certain objective, and the parameters needed to execute the function.
    """
)

LIBRARIAN_STANDARD_USER = dedent(
    """\
    -Goal-
    Given a description of the information that the user is seeking, and the list of resources/functions that can be used to gather the information, determine the functions that should be used to find the information and the parameters needed to execute the function.

    -Steps-
    1. For each resource/function, use your intuition and the function description to infer the type of information that the function can provide for the request.
    For each resource, determine the following information:
    - func_name: the function name in question. use the original name provided in the input
    - parameters: the parameters that will be entered to the function. It should be a stringified object.
    - purpose: describe what you want to learn by using this resource/function
    - rank: the importance of using this resource, where 1 is very important and must be used, and 3 is least important and can be ignored. Minimize the number of functions assigned with a rank of 1, unless absolutely necessary.

    Format each resource identification as (resc|<func_name>|<parameters>|<purpose>|<rank>|)

    2. Return output in English as a single list of all the resources identified in step 1, **ordered from most important to least important**, separated by a newline character.

    -Input-
    Description of the information that the user wants to know:
    {information_desc}

    Tools Descriptions in JSON Format:
    {resource_desc}

    -Output-
    """
)