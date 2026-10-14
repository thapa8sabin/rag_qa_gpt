from dotenv import load_dotenv
from generation.questiongenerator import QuestionGenerator
from generation.prompt_builder import generate_prompt
from src.retrieval.query_embedder import embed_query
from src.retrieval.reranker import reranker
from src.retrieval.retriever import retrieve

load_dotenv()

generator = QuestionGenerator()


def generate_questions(queries: str) -> list[dict[str, str]]:
    query_embedding = embed_query(queries)
    retrieved_docs = retrieve(query_embedding, top_k=10)
    reranked_docs = reranker(queries, retrieved_docs, top_k=8)
    context = "\n\n".join(reranked_result[0].payload["text"] for reranked_result in reranked_docs)
    result = generator.generate(article=context, use_evaluator=True, num_questions=5)
    return result


if __name__ == "__main__":
    questions = generate_questions("Generate questions and answers pairs from arts and litrature subject.")
    print(generate_prompt(questions))