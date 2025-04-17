PROMPTS = {}

# ------------ DEFAULT RAG PROMPTS ------------

PROMPTS["system_standard"] = "You are a helpful assistant."

PROMPTS["rag_prompt_standard"] = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\
 If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
 \nQuestion: {query}
 \nContext: {context}
"""

PROMPTS["rag_system_standard"] = "You are a helpful assistant that answers a query using the given contexts"

# ------------ CONTEXTUALIZE CHUNK PROMPTS ------------
# source: https://www.anthropic.com/news/contextual-retrieval
PROMPTS["contextualize_chunk_prompt"] = """<document> 
{WHOLE_DOCUMENT}
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{CHUNK_CONTENT}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


 # ------------ DOCUMENT/CHUNK INFORMATION EXTRACTION ------------
PROMPTS["extract_document_entities"] = """Given a text document, identify all entities of this document.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity.
- entity_type: One of the following types: [Firm, Industry, Country]
- entity_description: Short, succinct description of the entity's background and attributes\
 using only information from the provided text. Focus on areas related to the entity's business,\
 for example how it makes money, its business strength or weakness, its management style, its history, etc.
Format each entity as ("entity"|<entity_name>|<entity_type>|<entity_description>)

2. Return output in English as a single list of all entities identified in steps 1.

-Example-
Input:
<document>
...
</document>

Output:
("entity"|General Motors|Firm|General Motors is a major American automobile manufacturer known\
 for producing vehicles and automotive parts. It has been significantly impacted by US tariffs\
 on trading partners, causing its stock to fluctuate negatively when tariff news emerges and then\
 recover when those tariffs are postponed.)
("entity"|Ford|Firm|Ford is a prominent American automotive company tasked with manufacturing\
 vehicles and automotive components. The firm has experienced stock impacts due to the uncertainty\
 surrounding US tariffs on imports, similar to General Motors, suffering declines but showing\
 resilience with tariff postponement.)

-Real Data-
Input:
<document>
{WHOLE_DOCUMENT}
</document>

Output:
"""

PROMPTS["extract_chunk_events"] = """-Goal-
Given a chunk of a text document and a list of entities, identify\
 all events in the text chunk and determine the relationships between the events and the entities.

-Steps-
1. Identify all events. For each identified event, extract the following information:
- event_name: Name of the event.
- event_type: One of the following types: [Business Risk, Business Opportunity, Regulatory Risk, Regulatory Opportunity, Geopolitical Risk, Geopolitical Opportunity]
- event_description: Comprehensive description of the event's attributes and activities
Format each entity as ("event"|<event_name>|<event_type>|<event_description>)

2. For event identified in step 1 and entities data, identify all pairs of (event, entity).\
 For each pair of related entities and events, extract the following information:
- event_name: name of the event
- entity_name: name of the entity that is involved in the event. The entity could have caused the event, or be affected by the event.
- relationship_description: explanation to why you think the event and the entity are related.
- relationship_strength: a numeric score indicating strength of the relationship between the entity and the event. 1 indicates a weak relationship, 10 indicates a strong relationship.
Format each pair as ("relationship"|<event_name>|<entity_name>|<relationship_description>|<relationship_strength>)

4. Return output in English as a single list of all events and relationships identified in steps 1 and 2.

-Example-
Input:
<entities>
("entity"|Morgan Stanley Investment Management|Firm|A global financial services firm that provides\
 investment management and a range of advisory services, including asset management and financial advisory,\
 with a focus on enhancing the investment experience of its clients."),
 ("entity"|Diageo|Firm|A multinational beverage company known for its premium alcoholic drinks,\
 including spirits and beers, and recognized for its marketing strategies and brand portfolio."),
</entities>
<chunk>
...
</chunk>

Output:
("event"|HEDGE FUND SHORTING|Business Risk|The practice of hedge funds increasingly betting against certain stocks, particularly those exposed to tariff risks, reflecting investor sentiment towards potential losses in those sectors."),
("relationship"|HEDGE FUND SHORTING|Goldman Sachs|Goldman Sachs's research highlights the trend among hedge funds to short stocks exposed to tariff risks, indicating a direct observation and response to market conditions by the firm.|7),
("relationship"|HEDGE FUND SHORTING|Diageo|Diageo's stocks have seen a drop as a result of greater short selling by hedge funds, reflecting concerns over the beverage industry’s exposure to economic disruptions from tariffs.|7),

-Real Data-
Input:
<entities>
{ENTITIES}
</entities>
<chunk>
{CHUNK_CONTENT}
</chunk>

Output:
"""

PROMPTS["extract_questions"] = """-Goal-
Find all questions that this document can answer.\
 These questions will help retrieve the right information in a retrieval-augmented generation (RAG) system.

For each question, extract:
- question: A question that can be answered using only information from this document. Keep it general.
    - Good example: “How is Netflix affected by rising interest rates?”
    - Bad example: “What does the author Thomas Shelby think about interest rates and Netflix?”
- original_text: The exact part of the document that answers the question.
- category: The type of question. Acceptable categories are: [Current Event, Historical Event, Concepts, Analysis].\
 Please see the definition for each category below.

Definition of acceptable question categories:
- Current Event: Asking for facts (NOT analysis) about a recent event.
- Historical Event: Asking for facts (NOT analysis) about a past event.
- Concepts: Asking for general ideas, not specific facts or opinions. Example: “How does inflation affect tech companies?”
- Analysis: Asking for analysis or opinions on specific facts. Example: “Why did Congress pass the 99th Amendment forcing Americans to become cyborgs?”

Format output as follows:

QUESTION 1
question: [Insert question]
original_text: [Insert original text]
category: [Insert category]

QUESTION 2
question: [Insert question]
original_text: [Insert original text]
category: [Insert category]"""

# ------------ GENERATE SUB-QUESTIONS FROM QUERY ------------
PROMPTS["query_generate_questions"] = """-Goal-
Given a financial query, generate sub questions that, when answered, will help generate a more comprehensive answer.

-Steps-
1. Identify all entities that are referenced in the question. Extract the following factors:
- entity_name: Name of the entity
- entity_type: One of the following types: [Firm, Industry, Country, Individual]
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. For each entity classified as "firm", generate questions that will help build a better picture to understand the firm. For each question, produce the following:
- question_statement: the question itself
- question_type: the type of question. Valid question types are: ["Firm History", "Strategic Posture", "Market Environment", "Opportunities", "Threats", "Strengths", "Weaknesses", "Conceptual", "Street's View"]
Format each question as ("question"|<question_statement>|<question_type>)

3. Return output in English as a single list of all entities and questions identified in steps 1 and 2.

-Examples-
Query: How will firing nuclear weapons into Neptune affect diplomatic relationships between the United States Government and the Knights of the Sun?

Output:
(entity|United States Government|Country)
(entity|Knights of the Sun|Firm)
(question|How does the Knights of the Sun make money?|Firm History)

-Real Input-
Query: How will the repeal of the US anti-corruption law affect giant U.S. banks' (like JPMorgan) business practice?

Output:
"""