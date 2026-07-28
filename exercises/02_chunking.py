# Reasonable Predictions:
# 1. Ugliest worst-chunk: strategy (a) fixed-512 — it will slice the
#    rag-avanzato table mid-row, leaving cells with no header. --> TRUE 
# 2. Strategy (b) will still break convenzioni-reeno's bullet lists,
#    because sentence-packing doesn't know a bullet needs its intro line. --> FALSE 
# 3. Structure-aware (c) chunk count will be much lower, maybe 5-8 per article. --> TRUE 

# Final harvest: 
# worst chunk for each: 
# (a) the one truncating the table because headers are lost in the other one
# (b) final part in which more than a topic is covered and when embedding and doing the average that means a no-info vector
# (c) the big one, since it will cap token size of the model and 40% of information will be lost in the RAG

from pathlib import Path
import re
import statistics
from typing import Any

from sentence_transformers import SentenceTransformer

from raglab import chunk_structure, ChunkedDocs


DATA_DIR = Path("data/private")

def load_model(model = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model); 
    return model


def load_article(path: Path) -> tuple[str,str]:
    raw = path.read_text(encoding='utf-8')
    if raw.startswith('---'):
        text = raw.split("---", 2)[2]
    else: 
        text = raw
    title = path.stem
    return (title,text)

# (a)
def chunk_fixed(text: str, size: int = 512) -> list[str]: 
    chunked_text = [text[i:i+size] for i in range(0, len(text), size)]
    return chunked_text

# (b) cuts meaning to respect size
def chunk_overlap(text: str, size: int = 512, overlap: float = 0.15) -> list[str]: 
    """Sentence-packing chunker with whole-sentence overlap.

    Splits text into sentence atoms, then greedily packs them into chunks
    of at most `size` chars. Each chunk (except the first) is seeded with
    the trailing sentences of the previous chunk, up to `size * overlap`
    chars, so any idea straddling a boundary exists intact in at least
    one chunk.

    Contract / invariants:
    - Chunks are <= `size` chars, WITH TWO DOCUMENTED EXCEPTIONS below.
    - Overlap is made of whole sentences only, never character fragments
      (fragments would re-create the mid-thought cut this strategy exists
      to prevent).
    - The seed counts against the budget: `size` means size, so chunk
      length maps predictably onto the embedder's token window.
    - Every sentence of the input appears in at least one chunk
      (verified by reconstruction check).
    - The cursor advances on every pass: no input can hang this loop.

    Exceptions to the size promise:
    - An atom larger than `size` is emitted ALONE, unmodified: the size
      promise and the overlap guarantee are both waived at its boundaries.
      Better one visible monster chunk than a silent mangling.
    - An atom that fits `size` but not `size - seed` is packed with its
      seed: overlap kept, size promise bent by up to the seed length.

    Known limitations (findings from exercise 02, kept on purpose):
    - Sentence splitting assumes prose. Markdown tables, wikilink lists,
      and source blocks contain no sentence delimiters, so entire
      sections arrive as single oversized atoms (see Exhibit A: an
      876-char chunk fusing a table + related-articles + bibliography —
      three topics in one vector, embedding diluted to near-uselessness,
      brushing MiniLM's 256-token ceiling). No splitter fixes a false
      premise; structured markdown wants chunk_structure() instead.
    - Naive splitting still misfires on abbreviations ("e.g.", "et al.").
      Tokenizer-grade sentence segmentation deliberately out of scope.

    """
    # list of complete thoughts:
    #sentences:  [s1, s2, s3, s4, s5, s6, s7]
    chunked_sentences = re.split(r'(?<=\.)\s+', text)

    # chunk 1:    s1 s2 s3
    # chunk 2:       s3 s4 s5     ← s3 appears twice: that's the overlap
    # chunk 3:          s5 s6 s7  ← s5 again

    i = 0
    overlapped_chunks = []
    while i < len(chunked_sentences):
        start = i
        tmp = []
        if overlapped_chunks: 
            j = i-1
            while j >= 0 and sum(len(s) for s in tmp) < round(size * overlap):
                tmp.append(chunked_sentences[j])
                j -=1
            tmp.reverse()

        current = tmp
        budget = sum(len(s) for s in current)
        while i < len(chunked_sentences) and budget + len(chunked_sentences[i]) <= size: 
            current.append(chunked_sentences[i])
            budget += len(chunked_sentences[i])
            i +=1
        if i == start:
            if len(chunked_sentences[i]) > size: 
                current = [chunked_sentences[i]]
            else: 
                current.append(chunked_sentences[i])
            i +=1
        overlapped_chunks.append(" ".join(current))
    return overlapped_chunks



def report(name: str, chunks: list[str]) -> None: 

    count = len(chunks)
    chunk_lengths = [len(chunk) for chunk in chunks]
    min_len = min(chunk_lengths)
    max_len = max(chunk_lengths)
    median = statistics.median(chunk_lengths)

    print(f"Report for file: {name}\n\n")
    print(f"Count: {count}\n\nMin length: {min_len}\n\nMax length: {max_len}\n\nMedian length: {median}\n\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:\n\n{chunk}")
        print(f"{"─" * 40}\n\n")

    fat_chunk = max(chunks, key=len)
    model = load_model()
    print(f"Token for the fat chunk would be: {len(model.tokenizer.encode(fat_chunk))} in respect to 256 ceiling of the MiniLM")
    

def main():
    docs = []
    for path in DATA_DIR.glob("*.md"): 
        docs.append(load_article(path))
    
    for doc in docs: 
        #chunked_vector = chunk_fixed(doc[1])
        #report(doc[0], chunked_vector)

        #overlapped = chunk_overlap(doc[1])
        #report(doc[0], overlapped)

        structured = chunk_structure(doc[0], doc[1])
        report(doc[0], [c.text for c in structured])
        # Token for the fattest chunk (found in rag-avanzato) would be: 430. 174 tokens — 40% of the chunk — do not exist in the vector with current model






if __name__ == "__main__": 
    main()