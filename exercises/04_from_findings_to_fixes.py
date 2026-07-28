
import json

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


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
        

def main(): 
    eval_set = load_eval_set()
    children_collection = init_chroma()
    collection = init_chroma_parents()
    gold_ids = [x for e in eval_set for x in e['gold']]
    collection_ids = collection.get()["ids"]
    missing = set(gold_ids).difference(collection_ids)
    assert not missing, missing
    model = load_model()

    hits = []
    rrs = []
    # for n in [5,8,10,15]:
        # print(f"eval for n_results = {n}") -> best = 5 since there is a strong hit 
    for e in eval_set: 
        query_embedding = embed(e["query"], model)
        results = children_collection.query(query_embeddings = query_embedding, n_results = 5, include=["distances", "metadatas", "documents"])
        seen = set()
        deduped_parents = []
        # print(results["documents"][0][0])
        # print(results["metadatas"][0][0])
        # seen answers "have I seen this," deduped_parents answers "in what order did I encounter them"
        for child_id, metadata in zip(results["ids"][0], results["metadatas"][0]):
            parent = return_parent(metadata)
            if parent not in seen: 
                seen.add(parent)
                deduped_parents.append(parent)
            if len(deduped_parents) == 5: # only for eval reach at 5 for recall@5. In production, clear the similarity floor
                break

        sims = 1 - np.array(results["distances"][0])
        top1_sim = sims[0]
        if e["gold"]: 
            rr = reciprocal_rank(deduped_parents, e["gold"])
            hit = hit_at_k(deduped_parents, e["gold"])
            print(f"{e['query']} ({e['species']}): top1_sim={top1_sim:.3f}, hit: {hit}, rr: {rr}")
            hits.append(hit)
            rrs.append(rr)
        else:
            print(f"{e['query']} ({e['species']}, no-answer): top1_sim={top1_sim:.3f}")

    print(f"recall@5: {np.mean(hits)}\n\nMRR: {np.mean(rrs)}")




if __name__ == "__main__": 
    main()