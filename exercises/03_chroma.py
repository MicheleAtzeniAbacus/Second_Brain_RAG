from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import re

from raglab import chunk_structure, ChunkedDocs
from raglab.chunking import ChildChunk

DATA_DIR = Path("data/private")
SKIPPED_SECTIONS = {"Fonti", "Articoli correlati"}
THRESHOLD = 100 # grid-search: 1. chunk size splits 2. embed + indexing 3. retrieve 4. compute recall@5/MRR -> best one wins (doc depenendent)

def load_model(model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

def load_article(path: Path) -> tuple[str,str]:
    raw = path.read_text(encoding='utf-8')
    if raw.startswith('---'):
        text = raw.split("---", 2)[2]
    else: 
        text = raw
    title = path.stem
    return (title,text)

def prepare_data():
    model = load_model()
    docs = []
    structured_docs = []
    children_embeddings_docs = []
    children_docs = []
    for path in DATA_DIR.glob("*.md"): 
        docs.append(load_article(path))

    for doc in docs:
        children_list = []
        embeddings_list = []
        structured = chunk_structure(doc[0], doc[1])
        structured = [s for s in structured if s.section not in SKIPPED_SECTIONS] # filtering out not necessary sections
        for s in structured: 
            children = chunk_parent(s, model.tokenizer)
            texts = [child.text for child in children]
            embeddings = model.encode(texts, normalize_embeddings= True)
            children_list.extend(children)
            embeddings_list.extend(embeddings)
        structured_docs.append(structured)
        children_embeddings_docs.append(embeddings_list)
        children_docs.append(children_list)
    print(f"parent: {len(structured_docs[0])} vs children: {len(children_docs[0])}")
    return structured_docs, children_docs, children_embeddings_docs

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


def embed(text: str): 
    model = load_model()
    return model.encode(text, normalize_embeddings= True)

    
def init_chroma(): 
    client = chromadb.PersistentClient(path="index/")   # creates the dir, persists to disk
    collection = client.get_or_create_collection(
        name="vault", 
        metadata={"hnsw:space" : "cosine"}
    )
    return collection

def init_chroma_parents(): 
    client = chromadb.PersistentClient(path="index/")   # creates the dir, persists to disk
    collection = client.get_or_create_collection(
        name="vault_parents", 
        metadata={"hnsw:space" : "cosine"},
        embedding_function=None
    )
    return collection

def chunk_id(chunk: ChunkedDocs) -> str:
    return f"{chunk.article}::{chunk.section}"

def child_id(chunk: ChildChunk) -> str:
    return f"{chunk.parent}::{chunk.index}"

def add_data(collection: chromadb.Collection, structured_docs: list[ChunkedDocs]):
    for doc in structured_docs:
        collection.upsert(
            ids = [chunk_id(chunk) for chunk in doc], 
            documents = [chunk.text for chunk in doc], 
            metadatas = [{"article": chunk.article, "section": chunk.section, "strategy": "structure"} for chunk in doc]
        )

def add_children_data(collection: chromadb.Collection, children: list[ChildChunk], embeddings: list):
    for doc, embedding in zip(children, embeddings, strict=True):
        collection.upsert(
            ids = [child_id(child) for child in doc], 
            documents = [child.text for child in doc], 
            embeddings = embedding,
            metadatas = [{"parent": child.parent, "index": child.index} for child in doc]
        )


def build(collection: chromadb.Collection): 
    _ , children, embeddings = prepare_data()
    add_children_data(collection, children, embeddings)
    diff_and_prune_children(collection, children)

def build_parent(collection: chromadb.Collection): 
    parents , _, _ = prepare_data()
    add_data(collection, parents)
    diff_and_prune(collection, parents)

def diff_and_prune(collection: chromadb.Collection, structured_docs: list[ChunkedDocs]):
    current_ids = set(collection.get()["ids"])
    new_ids = {chunk_id(chunk) for doc in structured_docs for chunk in doc}
    stale_ids = current_ids - new_ids
    if stale_ids: 
        try:
            collection.delete(ids=list(stale_ids))
        except Exception as e:
            print(f"An error occured during vector store deletion:{str(e)}")
            return None

def diff_and_prune_children(collection: chromadb.Collection, children: list[ChildChunk]):
    current_ids = set(collection.get()["ids"])
    new_ids = {child_id(chunk) for doc in children for chunk in doc}
    stale_ids = current_ids - new_ids
    if stale_ids: 
        try:
            collection.delete(ids=list(stale_ids))
        except Exception as e:
            print(f"An error occured during vector store deletion:{str(e)}")
            return None

def chunk_parent(parent: ChunkedDocs, tokenizer, size: int = 100) -> list[ChildChunk]:
    body = parent.text.split("\n", 1)[1]
    parent_id = chunk_id(parent)
    if (len(tokenizer.encode(body)) <= THRESHOLD): 
        return [ChildChunk(parent=parent_id, index=0, text=body)]
    else:
        children_texts = chunk_overlap(body, size = size)
        children = []
        for i, child_text in enumerate(children_texts):
            children.append(ChildChunk(
                parent=parent_id,
                index = i, 
                text = child_text
            ))
        return children 

def build_all(): 
    parents_collection = init_chroma_parents()
    print(parents_collection.count())
    print(parents_collection.get()["ids"])
    # print(parents_collection.get(ids=["rag-avanzato::Punti chiave"], include=["documents", "metadatas"]))
    children_collection = init_chroma()
    print(children_collection.count())
    print(children_collection.get()["ids"])
    #print(children_collection.get(ids=["rag-avanzato::Punti chiave"], include=["documents", "metadatas"]))
    
    parents, children, embeddings = prepare_data()
    
    add_data(parents_collection, parents)
    diff_and_prune(parents_collection, parents)
    
    add_children_data(children_collection, children, embeddings)
    diff_and_prune_children(children_collection, children)

    child = children_collection.get(ids=["rag-avanzato::Punti chiave::3"], include=["metadatas"])
    parent_id = child["metadatas"][0]["parent"]
    print(parents_collection.get(ids = [parent_id]))


def main():
    build_all()
    # query_embedding = embed("What is Vertex AI?")
    # results = collection.query(query_embeddings = query_embedding, n_results = 5, include=["distances", "metadatas", "documents"])
    # print(results)

    # query_embedding = embed("Vertex AI Agent Builder RAG Engine Agent Garden DialogFlow NotebookLM")
    # results = collection.query(query_embeddings = query_embedding, n_results = 5, include=["distances", "metadatas", "documents"])
    # print(results)

    # small-to-big pattern check:
    # reduce dilution + no cut from embedder if chunk exceeds token size 
    # doc = load_article(Path('data/private/isolate-progetto-stage.md'))
    # parent_chunk = chunk_structure(doc[0], doc[1])
    # model = load_model()
    # children = chunk_parent(parent_chunk[0], model.tokenizer)
    # for c in children: 
    #     print(c.parent)
    #     print(c.index)
    #     print(len(c.text), len(model.tokenizer.encode(c.text)))
    #     print("\n")





    

    

if __name__ == "__main__": 
    main()


# Take home message: 
# 1. The whole result is noise floor. Convert: distances 0.72–0.83 → similarities 0.28–0.17. Exercise-01 relevant plateau lived at 0.54–0.59 similarity. Every single hit is junk-level. The profile is telling you "I have nothing" but  Chroma still returned five results. 
# That's the production lesson: without a distance gate, a RAG pipeline stuffs these five irrelevant chunks into the prompt and the LLM hallucinates an answer grounded in garbage. 
# This is precisely the hole CRAG's quality gate (pattern #6 in [[rag-pattern]]) exists to plug. 

# 2. What ranked: Articoli correlati, Fonti, Intro. 
# The winners are wikilink lists and source blocks 
# They rank because they're short and stuffed with "RAG / Google / agentic" and now they pollute every query.


#3. Finding of verbatim query confirms this. The "Vertex AI etc sentences" that were living in the big chunk that has been rncated 
# by the model embedder is not retrieved with a very low distance score. 


## Lab 1 vs Lab 2: 
#Head 0.35 vs tail 0.31. The gap between "verbatim text inside the window" and "verbatim text beyond it" is a measly 0.04. 
# Truncation is real — but it's not the dominant reason this chunk retrieves badly. The dominant reason is mean-pooling dilution. 
# That chunk is ~15 dense bullets on eight different topics: grounding, frozen RAG, failure modes, agentic RAG, four indexing techniques, latency numbers, Google products. 
# Its vector is the average of all of that — so your verbatim sentence is one voice in a fifteen-voice choir, and it only matches its own 1/15th of the blend.
# Document	Google content in vector	Score
# Full chunk (430 tok)	absent (amputated)	0.31
# Head only (256 tok)	absent	0.35
# Tail only (174 tok)	present, ~25% of the pool	0.41
# In a smaller and smaller chunk, becomes more and more findable:
# smaller denominator → your content owns more of the vector → higher score
# dose-response curve is measuring from the outside: 1/430ths → 1/174ths → 1/40ths of ownership.

# average is over token vectors: 
# Tokenize: your 430-token chunk → truncated to 256 token ids.
# Transformer: outputs one vector per token — 256 vectors, each 384-dimensional. 
# Mean pooling: take those 256 token vectors and average them, dimension by dimension — sum of 256 vectors / 256 — 
# into the single 384-dim vector that gets stored.

# truncation sets some contributions to exactly zero; dilution shrinks the rest by the denominator.
