
import json
import math

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_eval_set(): 
    with open("data/eval_queries.json") as f:
        return json.load(f)
    
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

def hit_at_k(parents: list[str], gold: list[str]) -> bool:
    for id in gold:
        if id in parents:
            return True
    return False
        
def reciprocal_rank(parents: list[str], gold: list[str]) -> float: 
    for rank, parent in enumerate(parents, start=1): 
        if parent in gold: 
            return 1/rank
    return 0.0

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

        

def main(): 
    eval_set = load_eval_set()
    children_collection = init_chroma()
    collection = init_chroma_parents()
    gold_ids = [x for e in eval_set for x in e['gold']]
    collection_ids = collection.get()["ids"]
    missing = set(gold_ids).difference(collection_ids)
    assert not missing, missing
    model = load_model()
    reranker = load_reranker()

    hits = []
    rrs = []
    # for n in [5,8,10,15]:
        # print(f"eval for n_results = {n}") -> best = 5 since there is a strong hit 
    for e in eval_set: 
        deduped_parents = search(e["query"], model, reranker, children_collection)
        if e["gold"]: 
            rr = reciprocal_rank(deduped_parents, e["gold"])
            hit = hit_at_k(deduped_parents, e["gold"])
            #print(f"{e['query']} ({e['species']}): top1_sim={top1_sim:.3f}, hit: {hit}, rr: {rr}")
            print(f"{e['query']} ({e['species']}): hit: {hit}, rr: {rr}, deduped parents: {deduped_parents}\n")
            hits.append(hit)
            rrs.append(rr)
        else:

            print(f"{e['query']} ({e['species']}): deduped parents: {deduped_parents} \n")

    # candidate_thresholds = [-3.00, -5.00, -5.5, -6.00, -7.00]
    # best_score = math.inf
    # best_t = None 
    # for t in candidate_thresholds:
    #     false_negatives = len([query for query in query_scores if query[1] and query[0] < t]) # genuine hit wrongly rejected
    #     false_positives = len([query for query in query_scores if not query[1] and query[0] >= t]) # no-answer wrongly accepted
    #     print(t, false_negatives, false_positives)
    #     cost = false_negatives + 2 * false_positives
    #     if cost < best_score: 
    #         best_score = cost
    #         best_t = t

    # best t = -6.00
    print(f"recall@5: {np.mean(hits)}\n\nMRR: {np.mean(rrs)}")




if __name__ == "__main__": 
    main()


## With re-ranking, queries that are not questions but keywords are going to "fail" in 
# finding sharp scores for answers due to the nature of the purpose. In such documents used in this demo. 
# indeed, there is a smooth decline of the scores rathen than a sharp deep.  