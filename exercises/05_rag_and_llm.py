


import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from raglab.answer import get_anthropic_client


def init_chroma(): 
    client = chromadb.PersistentClient(path="index/")   # creates the dir, persists to disk
    collection = client.get_collection(
        name="vault"
    )
    return collection

def init_chroma_parents(): 
    client = chromadb.PersistentClient(path="index/")   # creates the dir, persists to disk
    collection = client.get_collection(
        name="vault_parents", 
    )
    return collection

def embed(text: str, model: SentenceTransformer): 
    return model.encode(text, normalize_embeddings= True)

def load_model(model_name = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

def load_reranker(): 
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

def return_parent(metadata: dict) -> str:
    return metadata["parent"]

def search(query: str, model: SentenceTransformer, reranker: CrossEncoder,children_collection: chromadb.Collection, threshold: float = -6.00) -> list:
    query_embedding = embed(query, model)
    results = children_collection.query(query_embeddings = query_embedding, n_results = 15, include=["distances", "metadatas", "documents"])
    seen = set()
    deduped_parents = []

    ## here goes the re-rank logic using CrossEncoder
    ## scope: reads the query and passage together, in the same forward pass, 
    # and tries to judge "does this passage satisfy this question."
    raw_ids, raw_docs, raw_metadatas = results["ids"][0], results["documents"][0], results["metadatas"][0]
    pairs = [[query, doc] for doc in raw_docs]
    scores = reranker.predict(pairs)
    order = scores.argsort(descending=True)

    reordered_ids = [raw_ids[i] for i in order]
    reordered_metadatas = [raw_metadatas[i] for i in order]
    scores = [scores[i] for i in order]
    # print(results["documents"][0][0])
    # print(results["metadatas"][0][0])
    # sims = 1 - np.array(results["distances"][0])
    # top1_sim = sims[0]
    # seen answers "have I seen this," deduped_parents answers "in what order did I encounter them"
    for child_id, metadata, score in zip(reordered_ids, reordered_metadatas, scores):
        if score < threshold:
                break 
        parent = return_parent(metadata)
        if parent not in seen: 
            seen.add(parent)
            deduped_parents.append(parent)

    return deduped_parents

def chat_answer(query, model, reranker, children_collection, parents_collection, llm_client):
    deduped_parents = search(query, model, reranker, children_collection)
    
    if not deduped_parents:
        return "No relevant answer found in your notes."   # skip the LLM call entirely — nothing to ground it on
    
    parents = parents_collection.get(ids=deduped_parents)
    context = "\n\n".join(parents["documents"])
    
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so. Context: {context} Question: {query}"""

    response = llm_client.invoke(prompt)
    return response.content

def main():
    llm_client = get_anthropic_client()
    children_collection = init_chroma()
    collection = init_chroma_parents()
    model = load_model()
    reranker = load_reranker()

    #query = "What is SimpleBlocObserver?"
    query = "Is there any STAGE paper already published?"
    response = chat_answer(query, model, reranker, children_collection, collection, llm_client)
    print(f"Query:{query}\n")
    print(f"LLM response: {response}\n")

if __name__ == "__main__": 
    main()