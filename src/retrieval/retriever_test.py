from src.retrieval.query_embedder import embed_query
from src.retrieval.retriever  import retrieve
from src.retrieval.reranker import reranker

query  = "what is the HTTP "

#embed the query 
embedding = embed_query(query=query)
print("Query embedded")
print("Vector dimension:", len(embedding))

#retive candidate form the Qdrant 
results = retrieve(embedding,top_k=10)
print("\nQdrant results:")
print("Number of candidates:", len(results))


#Reranker 
reranker_result = reranker(
    query=query,
    results=results,
    top_k=3
)
print("\n\n========== RERANKED RESULTS ==========")


for result, score in reranker_result:
    print("\nReranker score:", score)
    print("Original Qdrant score:", result.score)
    print("Source:", result.payload["source_file"])
    print("Content:", result.payload["text"][:200])
