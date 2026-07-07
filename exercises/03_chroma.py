from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from raglab import chunk_structure, ChunkedDocs

DATA_DIR = Path("data/private")

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
    embeddings_docs = []
    for path in DATA_DIR.glob("*.md"): 
        docs.append(load_article(path))

    for doc in docs: 
        structured = chunk_structure(doc[0], doc[1])
        texts = [text.text for text in structured]
        embeddings = model.encode(texts, normalize_embeddings= True)
        structured_docs.append(structured)
        embeddings_docs.append(embeddings)

    
    return structured_docs, embeddings_docs

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

def add_data(collection: chromadb.Collection, structured_docs: list[ChunkedDocs], embeddings: list):
    for doc, embedding in zip(structured_docs, embeddings, strict=True):
        collection.upsert(
            ids = [f"{chunk.article}::{chunk.section}" for chunk in doc], 
            documents = [chunk.text for chunk in doc], 
            embeddings = embedding,
            metadatas = [{"article": chunk.article, "section": chunk.section, "strategy": "structure"} for chunk in doc]
        )

def build(collection: chromadb.Collection): 
    structured_docs, embeddings = prepare_data()
    add_data(collection, structured_docs, embeddings)

# def search(): 



def main():
    collection = init_chroma()
    #build()
    # print(collection.count())
    # print(collection.get(ids=["rag-avanzato::Punti chiave"], include=["documents", "metadatas"]))

    # query_embedding = embed("What is Vertex AI?")
    # results = collection.query(query_embeddings = query_embedding, n_results = 5, include=["distances", "metadatas", "documents"])
    # print(results)

    query_embedding = embed("Vertex AI Agent Builder RAG Engine Agent Garden DialogFlow NotebookLM")
    results = collection.query(query_embeddings = query_embedding, n_results = 5, include=["distances", "metadatas", "documents"])
    print(results)

    

    

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
