"""Assignment 3: The "Lost Context" Detective Puzzle.

A local RAG pipeline: RecursiveCharacterTextSplitter -> HuggingFace embeddings (run locally,
all-MiniLM-L6-v2) -> in-memory FAISS -> LLM. It first runs a small experiment showing which
chunk_size / chunk_overlap settings keep the clues together, then answers the question with
the chosen settings. See the explanation at the bottom of this file.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # langchain-community notice

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers.utils import logging as transformers_logging

from llm_setup import CHAT_MODEL, get_llm

transformers_logging.disable_progress_bar()  # hide the "Loading weights" bar

tricky_document = """
Section 1: The company Jade Global is launching a massive new
internal initiative called Project Phoenix. This project will
restructure the entire cloud infrastructure.
Section 2: Employees must adhere to the standard office hours of
9:00 AM to 5:00 PM. Remote work is permitted on Tuesdays and
Thursdays, provided that the employee has secured prior approval
from their direct manager.
Section 3: The cafeteria will now offer extended hours, opening
at 7:30 AM for breakfast. Please ensure you clear your tables
after eating.
Section 4: All IT support tickets must be filed through the
internal Jira portal. Direct emails to the IT staff will be
ignored starting next month.
Section 5: The annual holiday party is scheduled for December
15th. Dress code is semi-formal. Plus-ones are allowed if
registered by November 30th.
Section 6: Parking in the executive lot is strictly prohibited
for unauthorized vehicles. Violators will be towed at the owner's
expense.
Section 7: Health insurance open enrollment begins in October.
Please review the new dental and vision plans, as the providers
have changed this year.
Section 8: All employees must complete the mandatory
cybersecurity training module by the end of Q3. Failure to do so
will result in temporary suspension of VPN access.
Section 9: Regarding the cloud restructure initiative mentioned
earlier, the final deadline for its completion is December 31st,
2026. The budget approved is $500,000.
"""

QUESTION = "What is the deadline and budget for Project Phoenix?"
EXPECTED = "December 31st, 2026, with a $500,000 budget"

# Final settings (see the explanation at the bottom of this file)
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 3  # chunks retrieved per question

# Settings tried during the experiment
EXPERIMENTS = [(100, 0), (150, 0), (250, 0), (250, 50)]

# The clues the retrieved context must contain for the LLM to connect the dots
CLUES = {
    "Project Phoenix": "Project Phoenix",                        # the name (Section 1)
    "Phoenix = cloud restructure": "restructure the entire cloud",  # what it is (Section 1)
    "cloud restructure initiative": "cloud restructure initiative",  # Section 9's subject
    "full deadline": "December 31st, 2026",                         # Section 9
    "budget": "$500,000",                                           # Section 9
}

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(tricky_document)
    store = FAISS.from_texts(chunks, embeddings)
    return chunks, store.as_retriever(search_kwargs={"k": TOP_K})


def flatten(text):
    return " ".join(text.split())  # join the document's line breaks into single spaces


def run_experiments():
    print(f"Experiment: which settings keep every clue in the top {TOP_K} retrieved chunks?\n")
    print(f"{'chunk_size':>10} {'overlap':>8} {'chunks':>7}  {'result':<8} missing clues")
    for size, overlap in EXPERIMENTS:
        chunks, retriever = build_retriever(size, overlap)
        context = flatten(" ".join(d.page_content for d in retriever.invoke(QUESTION)))
        missing = [name for name, text in CLUES.items() if text not in context]
        print(f"{size:>10} {overlap:>8} {len(chunks):>7}  {'OK' if not missing else 'BROKEN':<8} "
              f"{', '.join(missing) or '-'}")
        section9 = [flatten(c) for c in chunks if "budget" in c]
        print(f"{'':>28}chunk with the budget: \"{section9[0]}\"")


answer_prompt = PromptTemplate.from_template(
    """Answer the question using ONLY the context below. The context comes from different
parts of one document, and the same project may be referred to by different names in
different parts, so link them using what the context says. If the answer is not in the
context, say "I don't know".

Context:
{context}

Question: {question}

Answer in exactly this format and nothing else:
<full deadline date, including the day, month and year>, with a <budget> budget"""
)


def answer_question():
    chunks, retriever = build_retriever(CHUNK_SIZE, CHUNK_OVERLAP)
    docs = retriever.invoke(QUESTION)

    print(f"\nFinal settings: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, "
          f"top {TOP_K} chunks ({len(chunks)} chunks total)")
    print(f"Question: {QUESTION}")
    print("Retrieved chunks:")
    for i, doc in enumerate(docs, start=1):
        print(f"  [{i}] {flatten(doc.page_content)}")

    chain = answer_prompt | get_llm(temperature=0) | StrOutputParser()
    # Join each chunk's line breaks so values split across lines (e.g. "December 31st,"
    # / "2026.") reach the LLM as one piece of text
    answer = chain.invoke({"context": "\n\n".join(flatten(d.page_content) for d in docs),
                           "question": QUESTION}).strip()
    print(f"\nAnswer ({CHAT_MODEL}): {answer}")
    correct = "December 31st, 2026" in answer and "$500,000" in answer
    print(f"Correct (expected \"{EXPECTED}\"): {correct}")


if __name__ == "__main__":
    run_experiments()
    answer_question()


'''
WHY chunk_size=250 AND chunk_overlap=50 (plus retrieving the top 3 chunks)
===========================================================================

The puzzle: the name "Project Phoenix" appears only in Section 1, while the deadline and
budget appear only in Section 9, which calls it "the cloud restructure initiative". The
only thing linking the two is the cloud-restructuring wording: Section 1 says Project
Phoenix "will restructure the entire cloud infrastructure", and Section 9 talks about "the
cloud restructure initiative". The LLM can only connect the dots if (a) the Section 1 chunk
keeps "Project Phoenix" together with "restructure the entire cloud", (b) the Section 9
chunk keeps "cloud restructure initiative" together with the deadline and budget, and
(c) both chunks are retrieved. The experiment at the top of this script checks exactly this.

An important point: no overlap setting can glue Section 1 to Section 9 directly. They are
about 1,200 characters apart, and overlap only repeats text across the boundary between
two NEIGHBOURING chunks. Overlap's real job here is to stop a single section from being
torn in half. Retrieving several chunks (k=3) is what brings Section 1 and Section 9 into
the same prompt.

What happened with bad settings:
- chunk_size=100, overlap=0: every section is shredded into ~2-3 fragments (22 chunks).
  Section 9 becomes three pieces: "Section 9: Regarding the cloud restructure initiative
  mentioned" / "earlier, the final deadline for its completion is December 31st," /
  "2026. The budget approved is $500,000." The budget fragment does not say WHICH
  initiative it belongs to, the year is cut off from "December 31st", and "Project Phoenix"
  is separated from "restructure the entire cloud infrastructure". This is context
  fragmentation: the retriever returns pieces that mean nothing on their own.
- chunk_size=150, overlap=0: better, but Section 9 is still split, so the chunk holding the
  deadline and budget has lost its subject.
- chunk_size=250, overlap=0: this is the exact failure the assignment describes. The
  chunks are big enough to hold a section, but with no overlap the boundaries fall in bad
  places: the splitter packs Section 8 plus the start of Section 9 into one chunk, ending
  with "...Section 9: Regarding the cloud restructure initiative mentioned", and the next
  chunk starts "earlier, the final deadline ... December 31st, 2026. The budget approved is
  $500,000." The deadline and budget are retrieved, but the phrase saying they belong to
  the cloud restructure initiative is stranded in a different chunk, so the LLM cannot
  tell that they belong to Project Phoenix.

Why the final numbers fix it:
- chunk_size=250: the longest section in the document is 216 characters (Section 2), and
  Section 9 is 166, so 250 is large enough for any single section to fit whole in one
  chunk, with a little room to spare. It is still small enough that each chunk is about
  one topic, so the embeddings stay precise and retrieval does not drag in the whole
  document. (Putting the entire 1,452-character document in one chunk also "works", but
  that is no longer RAG: every question would get every section, including all the noise.)
- chunk_overlap=50: when a chunk boundary falls in the middle of a section, the next chunk
  repeats the last ~50 characters of the previous one. That repeated text is exactly
  "Section 9: Regarding the cloud restructure initiative mentioned", so Section 9 now
  appears whole in one chunk: its subject, deadline and budget stay together. 50 is about
  one line of this document (lines are ~60-65 characters), which is enough to carry a
  section's opening line across a boundary without duplicating large amounts of text.
- Top 3 chunks: the question mentions "Project Phoenix" (matching the Section 1 chunk) and
  "deadline and budget" (matching the Section 9 chunk), so both are retrieved together,
  and the prompt tells the LLM that the same project may be named differently in
  different parts. With both intact chunks in front of it, the LLM links "Project Phoenix
  ... restructure the entire cloud infrastructure" to "the cloud restructure initiative"
  and answers: December 31st, 2026, with a $500,000 budget.
'''
