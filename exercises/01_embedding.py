from sentence_transformers import SentenceTransformer
import numpy as np

demo_data = [
    # RAG — from wiki/project-daedalus
    ("rag", "Semantic chunking detects when the topic shifts and breaks there, instead of splitting every n tokens."),
    ("rag", "HyDE first generates a hypothetical answer, then retrieves using that as the embedding, because document and query embeddings differ in structure but not in content."),
    ("rag", "Contextual Retrieval rewrites each chunk with document context before embedding, cutting failed retrievals by 49 percent."),

    # Flutter — from wiki/reeno-app
    ("flutter", "Use Cubit for simple linear state transitions like auth status or connectivity, and Bloc when there are multiple event types or complex branching."),
    ("flutter", "States extend Equatable; prefer a single state class with a status enum over subclasses, and never store mutable objects in state."),
    ("flutter", "Screen-local BLoCs are created in initState and disposed in dispose, while app-global BLoCs live in the main provider tree."),

    # STAGE — from wiki/stage
    ("stage", "STAGE is an EU project on healthy ageing organized in ten work packages over a 72-month timeline from 2024 to 2029."),
    ("stage", "A cross-work-package group ensures ethical and practical recommendations are embedded in the tools being developed, avoiding contradictory outputs."),
    ("stage", "Ethical review is most useful early in development, not as an end-stage constraint on a finished platform."),

    # Admin — from wiki/documenti-personali (catalog-level)
    ("admin", "The mail backup folder contains four Outlook archives that are restorable in Outlook but not parsed as knowledge."),
    ("admin", "The icons folder holds ten SVG interface icons that look like a navigation bar asset set."),
    ("admin", "An Excalidraw drawing stores a compressed JSON payload with no text elements and is unreadable outside Obsidian."),
]

traps = [('trap', 'Flutter widgets rebuild whenever the state changes.'), ('trap', 'The state reimburses travel expenses after the claim is approved.')]


def load_model(model = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model); 
    return model


def embed(model: SentenceTransformer, corpus: list[tuple[str]]) -> np.ndarray: 
    text = [text for _, text in corpus]
    embeddings = model.encode(text, normalize_embeddings= True)
    ##  Every value is around ±0.05. 
    # If a vector is normalized (length 1) and that length is spread across 384 dimensions, 
    # the typical component size is about 1/√384 ≈ 0.051. 
    #print(np.linalg.norm(embeddings))
    return embeddings

def get_similarity_matrix(embeddings: np.ndarray): 
    matrix = embeddings @ embeddings.T 
    return matrix

def retrieve(queries: np.ndarray, corpus_embeddings: np.ndarray):
    return queries @ corpus_embeddings.T 


def main(): 
    model = load_model()
    corpus = demo_data + traps
    labels = [label for label, _ in corpus]
    texts  = [text for _, text in corpus]
    embeddings = embed(model, corpus)
    corpus_matrix = get_similarity_matrix(embeddings=embeddings)
    print(corpus_matrix.shape)
    query_corpus = [ 
        ("q1", "how do I split documents for embedding?"),
        ("q2", "how does the app manage state between screens?")
    ]
    queries = embed(model, query_corpus)
    retrieved_matrix = retrieve(queries, embeddings)

    for (q_label, q_text), row in zip(query_corpus, retrieved_matrix):
        print(f"\nQuery: {q_text}")
        for rank in np.argsort(row)[::-1]:
            print(f"  {row[rank]:+.2f}  {labels[rank]:8}  {texts[rank][:60]}")
        


    # for sentence 1 [2, 0, 1, ...]. index 0 was supposed to win on pure meaning 
    # instead it got pipped by the contextual-retrieval sentence. 
    # Why? That sentence contains the literal word "embedding", and the query does too. 
    # Lesson: embeddings are not immune to surface vocabulary — lexical overlap still nudges the geometry. 
    # The clean story "keyword search = words, semantic search = meaning" is a simplification; 
    # in reality embedders blend both, which is why hybrid search helps rather than duplicates.


    # for sentence #2 [5, 12, 4, ...., 13] sentence 13, "the state reimburses travel expenses" 
    # landed at rank 10, below every Flutter sentence despite sharing the word "state". 
    # That's the cleanest result of the exercise: the model resolved the polysemy of "state" from context. 

if __name__ == "__main__": 
    main()
