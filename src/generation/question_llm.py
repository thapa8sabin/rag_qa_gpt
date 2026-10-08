from dotenv import load_dotenv
from generation.questiongenerator import QuestionGenerator
from src.retrieval.query_embedder import embed_query
from src.retrieval.reranker import reranker
from src.retrieval.retriever import retrieve

load_dotenv()

generator = QuestionGenerator()

def generate_questions(queries: str) -> list[dict[str, str]]:
    query_embedding = embed_query(queries)
    retrieved_docs = retrieve(query_embedding, top_k=8)
    reranked_docs = reranker(queries, retrieved_docs, top_k=10)
    context = "\n\n".join(
        reranked_result[0].payload["text"] for reranked_result in reranked_docs
    )
    result = generator.generate(article=context, num_questions=5)
    return result
