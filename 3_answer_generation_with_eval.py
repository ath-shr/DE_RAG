import os
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", api_key=os.getenv("NVIDIA_API_KEY"))

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query  = input("Enter your query: ") #= "What problem does KG-RAG solve?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatNVIDIA model
# model = ChatNVIDIA(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)



# ----------------------------------------------------
# Evaluate retrival

eval_system_prompt = """
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate the generated response using ONLY:
1. The user query
2. The retrieved documents/context
3. The generated response

Do NOT use external knowledge or your own knowledge to fill missing information.

Evaluate the generated response on the following four dimensions.

1. ACCURACY
Determine whether the factual claims in the generated response correctly answer the user's query and are supported by the retrieved context.

Scoring:
- 10: Completely correct; no factual errors.
- 8-9: Essentially correct with only minor imprecision.
- 6-7: Mostly correct but contains a minor factual error or ambiguity.
- 4-5: Partially correct; contains significant errors or misses an important aspect.
- 2-3: Mostly incorrect.
- 1: Completely incorrect or contradicts the retrieved context.

2. COMPLETENESS
Determine whether the generated response addresses all aspects of the user's query that can be answered using the retrieved context.

Scoring:
- 10: Fully answers the query and includes all important information available in the context.
- 8-9: Answers the query completely with only minor omissions.
- 6-7: Answers the main question but omits some relevant information.
- 4-5: Provides only a partial answer.
- 2-3: Addresses very little of the query.
- 1: Does not answer the query.

IMPORTANT:
Do not penalize the response for information that is not present in the retrieved context.

3. FAITHFULNESS
Determine whether every factual claim in the generated response is supported by the retrieved context.

A response is faithful when its claims can be directly supported or reasonably inferred from the retrieved documents.

Penalize:
- Hallucinated facts
- Information not present in the retrieved context
- Claims that contradict the retrieved context
- Unsupported assumptions presented as facts

Scoring:
- 10: Every factual claim is supported by the retrieved context.
- 8-9: Nearly all claims are supported; only very minor unsupported details.
- 6-7: Mostly supported but contains one or more unsupported claims.
- 4-5: Contains several unsupported claims.
- 2-3: Large portions are unsupported.
- 1: The response is predominantly hallucinated or contradicts the context.

4. OVERALL QUALITY
Evaluate the overall usefulness of the generated response considering:
- Accuracy
- Completeness
- Faithfulness
- Relevance to the query
- Clarity and conciseness

Do not reward a response simply because it is well-written if its facts are incorrect or unsupported.

Scoring:
- 10: Excellent response; correct, complete, faithful, relevant, and clear.
- 8-9: Very good response with minor issues.
- 6-7: Acceptable response but has noticeable issues.
- 4-5: Poor response with significant problems.
- 2-3: Very poor response.
- 1: Completely unusable.

IMPORTANT EVALUATION RULES:
- Evaluate the answer against the retrieved context, not against external knowledge.
- Do not assume that a statement is true simply because you know it to be true.
- If the retrieved context does not contain enough information to answer the query, a response that explicitly says the information is unavailable should NOT be penalized for incompleteness.
- If the generated response gives an answer that is not supported by the retrieved context, reduce the faithfulness score even if the answer may be factually true in the real world.
- If the generated response contradicts the retrieved context, strongly reduce accuracy and faithfulness.
- Do not penalize harmless differences in wording, formatting, or numerical representation when the meaning is equivalent.
- Base scores on evidence, not subjective preference.
- Be strict and consistent across evaluations.

Return ONLY valid JSON. Do not include markdown, code fences, or additional text.

Use exactly this JSON structure:

{
  "accuracy": {
    "score": <integer from 1 to 10>,
    "justification": "<brief evidence-based justification>"
  },
  "completeness": {
    "score": <integer from 1 to 10>,
    "justification": "<brief evidence-based justification>"
  },
  "faithfulness": {
    "score": <integer from 1 to 10>,
    "justification": "<brief evidence-based justification>"
  },
  "overall_quality": {
    "score": <integer from 1 to 10>,
    "justification": "<brief evidence-based justification>"
  }
}
"""

combined_eval_input = f"""
<evaluate_retrieval>
    <query> 
        {query} 
    </query>
    <retrieved_documents>
        {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
    </retrieved_documents>
    <generated_response>
        {result.content}
    </generated_response>
</evaluate_retrieval>
"""


eval_model = ChatNVIDIA(model="openai/gpt-oss-20b")
eval_messages = [
    SystemMessage(content=eval_system_prompt),
    HumanMessage(content=combined_eval_input),
]

eval_result = eval_model.invoke(eval_messages)

print(f"Eval Result: {eval_result.content}")