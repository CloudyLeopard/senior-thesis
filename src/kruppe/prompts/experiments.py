from textwrap import dedent

GENERATE_QUESTION_SYSTEM = dedent(
    """\
    You are a helpful assistant that extracts the central business question a document seeks to answer.
    """
)

GENERATE_QUESTION_USER = dedent(
    """\
    -Instruction-
    Given a document, identify the main, high-level question that the document seeks to answer or address. Answer only with the question.

    -Input-
    <document>
    {document}
    </document>

    -Output-
    """
)

CATEGORIZE_QUESTION_USER_CHAIN = [
    dedent(
        """\
        -Instruction-
        The following is a list of questions that are addressed in different documents (e.g. research reports and news articles) from the same industry. Please analyze the common categories that you find arising among the questions. Categories should be mutually exclusive from one another. Respond only with a numbered list, ranking from most common to least common. 

        -Input-
        <questions>
        {numbered_questions}
        </questions>

        -Output-
        """
    ),
    dedent(
        """\
        -Instruction-
        Using the output categories, categorize each of the numbered questions with their corresponding category. Only one category should be assigned to each question. Respond only with a numbered list, where each number corresponds to the question's number.

        -Output-
        """
    )
]

VANILLA_QA_SYSTEM = dedent(
    """\
    You answer research questions. Format your response like a report, such as have a central thesis, use supporting arguments and backup you points.
    """
)

# based off LightRAG's evaluation prompt
EVALUATE_REPORT_SYSTEM = dedent(
    """\
    -Role-
    You are an expert taked with evaluating two reports that seek to answer the same question based on five criteria: comprehensiveness, diversity, empowerment, and cohesiveness.

    -Goal-
    You will evaluate two answers to the same question based on four criteria: comprehensiveness, diversity, empowerment, and cohesiveness.

    - Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
    - Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
    - Empowerment: How well does the answer help the reader understand and make informed judgements about the topic?

    For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.
    """
)

EVALUATE_REPORT_USER = dedent(
    """\
    Here is the question: {query}
    
    Here are the two answers:
    <answer1>
    {answer1}
    </answer1>

    <answer2>
    {answer2}
    </answer2>

    Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.

    Output your evaluation in the following JSON format (but in string form):

    {{
        "Comprehensiveness": {{ 
            "Winner": "[Answer 1 or Answer 2]", 
            "Explanation": "[Provide explanation here]" 
        }},
        "Diversity": {{ 
            "Winner": "[Answer 1 or Answer 2]", 
            "Explanation": "[Provide explanation here]" 
        }},
        "Empowerment": {{ 
            "Winner": "[Answer 1 or Answer 2]", 
            "Explanation": "[Provide explanation here]" 
        }},
        "Cohesiveness": {{ 
            "Winner": "[Answer 1 or Answer 2]", 
            "Explanation": "[Provide explanation here]" 
        }},
        "Overall Winner": {{ 
            "Winner": "[Answer 1 or Answer 2]", 
            "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]" 
        }}
    }}
    """
)