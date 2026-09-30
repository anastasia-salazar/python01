"""Assignment 2: The "Smart Splitter" Proof.

Splits long_document.txt with RecursiveCharacterTextSplitter (chunk_size=200,
chunk_overlap=50), then proves with a custom validation function that consecutive chunks
really share overlapping text.

Why the overlap is "up to" 50 characters rather than exactly 50: the splitter never cuts
words in half. It carries over as many whole words from the end of a chunk as fit in 50
characters, so the measured overlap is usually a bit under 50. It also doesn't overlap
across paragraph breaks, since each paragraph is split separately first.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
DOCUMENT = Path(__file__).resolve().parent / "long_document.txt"


def validate_overlap(chunk_a, chunk_b, original_text, max_overlap=CHUNK_OVERLAP):
    """Prove two consecutive chunks overlap, and extract the shared text.

    1. Text check: find the longest string that is BOTH the end of chunk A and the start
       of chunk B.
    2. Position check: using each chunk's start position in the original document,
       confirm those characters are the very same span of the original text (not just a
       coincidentally repeated phrase).
    """
    text_a, text_b = chunk_a.page_content, chunk_b.page_content

    overlap = ""
    for length in range(min(len(text_a), len(text_b)), 0, -1):
        if text_a[-length:] == text_b[:length]:
            overlap = text_a[-length:]
            break

    start_a, start_b = chunk_a.metadata["start_index"], chunk_b.metadata["start_index"]
    end_a = start_a + len(text_a)
    span_in_original = original_text[start_b:end_a] if start_b < end_a else ""

    return {
        "overlap": overlap,
        "length": len(overlap),
        "same_span_in_original": bool(overlap) and span_in_original == overlap,
        "within_limit": len(overlap) <= max_overlap,
    }


def main():
    text = DOCUMENT.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # record where each chunk starts in the original text
    )
    chunks = splitter.create_documents([text])

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    print(f"Document: {len(text)} characters, {len(paragraphs)} paragraphs")
    print(f"Splitter: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP} "
          f"-> {len(chunks)} chunks\n")

    # ---- The required output: first two chunks and their extracted overlap ----
    first = validate_overlap(chunks[0], chunks[1], text)
    print("=" * 80)
    print(f"Chunk 1 ({len(chunks[0].page_content)} chars):\n{chunks[0].page_content}\n")
    print(f"Chunk 2 ({len(chunks[1].page_content)} chars):\n{chunks[1].page_content}\n")
    print(f'Extracted Overlap ({first["length"]} chars): "{first["overlap"]}"')
    print(f"  Same text at the same position in the original document: "
          f"{first['same_span_in_original']}")
    print(f"  Within the {CHUNK_OVERLAP}-character limit: {first['within_limit']}")
    print("=" * 80)

    # ---- Validate every consecutive pair ----
    print("\nAll consecutive pairs:")
    print(f"{'Pair':<9}{'Overlap':>8}  {'Verified':<10}Overlapping text")
    for i in range(len(chunks) - 1):
        result = validate_overlap(chunks[i], chunks[i + 1], text)
        if result["overlap"]:
            status = "yes" if result["same_span_in_original"] and result["within_limit"] else "NO"
            shown = f'"{result["overlap"]}"'
        else:
            status = "-"
            end_a = chunks[i].metadata["start_index"] + len(chunks[i].page_content)
            gap = text[end_a:chunks[i + 1].metadata["start_index"]]
            shown = ("(none: the next chunk starts a new paragraph)" if "\n\n" in gap
                     else "(none: NO OVERLAP FOUND)")
        print(f"{i + 1:>2} -> {i + 2:<3}{result['length']:>8}  {status:<10}{shown}")


if __name__ == "__main__":
    main()
