import pytest
import pytest_asyncio
from aiohttp import ClientSession

from rag.models import Document

@pytest_asyncio.fixture(scope="session")
async def session():
    client = ClientSession()
    yield client
    await client.close()
    
@pytest.fixture
def texts():
    return [
        "The transactions are relatively small for now. Still, they are intertwining banks (in Wall Street parlance, the sell side) with investors (the buy side) in ways that are new and difficult to parse for analysts, regulators and others.",
        "“There’s a lot of scrutiny on the potential for spillover of exposure from private credit into the broader banking system,” said Roy Choudhury, a managing director at Boston Consulting Group who advises banks on business with private-fund managers.",
    ]

@pytest.fixture
def documents():
    doc1 = Document(
        text="""Retrieval augmented generation (RAG) is a technique that grants generative artificial intelligence models information retrieval capabilities. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information to augment information drawn from its own vast, static training data. This allows LLMs to use domain-specific and/or updated information. Use cases include providing chatbot access to internal company data, or giving factual information only from an authoritative source.""",
        metadata={
            "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "source": "wikipedia",
        },
    )

    doc2 = Document(
        text="""Goldman Sachs GS -0.21% decrease; red down pointing triangle this month sold $475 million of public asset-backed securitization, or ABS, bonds backed by loans the bank makes to fund managers that tide them over until cash from investors comes in. The first-of-its-kind deal is a lucrative byproduct of the New York bank’s push into loans to investment firms, such as these so-called capital-call lines.

Goldman’s new deal reflects two trends transforming financial markets. Increasingly large managers of private-debt and private-equity funds are moving up in the Wall Street pecking order, but they often need money fast. Banks, once again, are reinventing themselves to adapt.

Bankers say the capital-call ABS and similar innovations help them safely serve clients while bringing in rich fees. But such efforts have preceded market excess in the past, to put it mildly. Skeptics see parallels between CDOs (the collateralized debt obligations that helped fuel the financial crisis in 2008) and the growing use of SRTs (synthetic risk transfers), NAV loans (based on net asset values) and more.""",
        metadata = {
            "url": "https://www.wsj.com/finance/investing/watch-out-wall-street-is-finding-new-ways-to-slice-and-dice-loans-d80415dc?mod=finance_lead_pos2",
            "source": "wall street journal"
        }
    )
    return [doc1, doc2]
