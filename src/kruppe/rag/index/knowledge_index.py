"""
Experiment on creating a "knowledge index" with the following idea

Suppose we treat a news article as effectively a combination of
1. fact [required]: the subject or event that prompted the author to write about it.
    e.g. macro changes, business operation changes
2. analysis [optioinal]: the author's analysis of said subject or event. Contains the author's
    *opinion* and/or *line of reasoning*
3. evidence [optional]: events in the past that support the author's line of reasoning,
    analysis by otheres that the author is citing, etc.

Given this idea, suppose we extract these three piece of information from a news article (but
also linking them together in one "Document" object).

Our task is, given a financial query, we want to provide a answer that is comprehensive
but also trustworthy. I've noticed an equity research report, which is a REALLY good
example of what a good answer looks like, contains the following elements:
1. fact - same definition as above
2. company story: business operation, description, management style, etc.
3. analysis (qualitative)
4. analysis (quantitative)
5. reference to other facts or past events and their affects
6. streets view - reference to how other people perceive said event

We do not trust LLMs to directly generate an answer to a query, because the query can be very layered,
and LLMs may not be able to think too deeply into the cause and effects. But, simple retrieval
may not be able to get us what we want - what if we just retrieve "facts"? What if we don't retrieve the
"right information"?

MY IDEA: create a vector storage (index) that *only* stores the facts. Each fact is linked to all
analysis and evidence associated with it. Then, given a query, we can retrieve the top most relevant
past events as well as people's analysis of said events. Using these information, we can generate
a very comphrensive answer.

The retriever should be more accurate too. Suppose we have a query: "How will netflix changing to
ad-supported model affect its stock price?" Now, since the vectorstorage is only storing facts, not only
will we retrieve analysis on people who wrote about this business operation change, but will also retrieve
past events of *other companies* changing their business model, and what happens! Since,
- "netflix changing to ad-supported model" and
- "amazon changing to ad-supported model"
will be semantically similar

Idea: need to include a "time" factor. Recent news are for "facts", while really old news
should be more for "analysis" to draw parallel.

Idea: to get information from both the story itself, but also specific to the firm itself, we
may want to use both semantic retrieval (vector search) and keyword retrieval (e.g. bm25)

Idea: the model should have "deduplication" feature. As we add more documents to it, there may be
several "facts" or "events" that are similar to the new ones already in the storage unit. Then,
we should combine them into one, and put all the analysis behind that one event.

Idea: calling on wikipedia database (with embedding) to get more information on an event.

Extension: using LLM, we can even generate *different* perspectives by identifying diverging
analysis/opinions *on the same fact*.

Output:
"""


"""
PART 2: Extraction of Information

Idea 1: Pass entire document into LLM, and ask it to return a JSON data of information I want
Pro: easy to implement
Con: LLM may not be able to extract the information, or may be too slow.

Idea 2: What about... using a method similar to "contextual retrieval"?
We first will still pass the entire document into the llm to extract the main subject
and the author's main analysis. Afterwards, we will use the "contextual retrieval" method
1. chunk the documents
2. pass each chunk into the llm ALONG WITH THE ORIGINAL DOCUMENT as the context
3. using that as context, extract the information we want
"""

"""
PART 3: Generation of Answer

Idea: the answer should be like a 'story', see how good equity research reports are formatted
"""