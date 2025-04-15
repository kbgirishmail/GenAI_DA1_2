import time
import pandas as pd
from app import get_rag_response, embedding_model, llm, collection # Import necessary components

# --- Configuration ---
TEST_QUESTIONS_FILE = None # Set to a file path if loading questions from CSV/JSON
OUTPUT_FILE = "evaluation_results.csv"
API_CALL_DELAY = 15

# Define test questions
TEST_QUESTIONS = [
    "How do I set up a new account?", # Simple retrieval, basic RAG functionality
    "What forms of payment are accepted?", # Simple retrieval, basic RAG functionality
    "Tell me the steps to track my order.", # Simple retrieval, basic RAG functionality
    "What is your policy on returning items?", # Simple retrieval, basic RAG functionality
    "Is it possible to cancel an order I just placed?", # Simple retrieval, basic RAG functionality
    "Within how many days can I return a product for a full refund?", # Detail-oriented, tests precision
    "What is the typical delivery time for standard shipping?", # Detail-oriented, tests precision
    "To which countries do you offer international shipping?", # Detail-oriented, tests faithfulness (needs to include "select countries")
    "What information do I need to provide to change my shipping address?", # Detail-oriented, tests extraction of specific details
    "What are the two ways I can get in touch with customer support?", # Detail-oriented, tests extraction of multiple pieces of information
    "Can I get a full refund if I return an item after 35 days?", # Harder, tests faithfulness to the return policy (should contradict the 30-day limit)
    "If my package is lost, what is the first step I should take?", # Detail-oriented, tests for a specific action
    "Is there a fee for your gift wrapping service?", # Simple retrieval of a specific detail
    "Will you match the price of a product I found cheaper on any website?", # Harder, tests faithfulness to the specific price matching policy (competitor's website)
    "Can I place an order by calling your phone number?", # Simple retrieval of a negative constraint
    "I received the wrong item. What should I do?", # Simple retrieval of a specific procedure
    "Can I use two discount codes on a single purchase?", # Simple retrieval of a constraint
    "What happens if a product I bought last week is now on sale?", # Detail-oriented, tests understanding of the price adjustment policy timeframe
    "If I don't create an account, can I still buy something?", # Simple retrieval of a specific option
    "Do you offer discounts for large quantity orders?" # Simple retrieval of a specific service/policy
]

# --- Evaluation Functions (Using LLM-as-Judge - simplified example) ---

def evaluate_faithfulness(question, answer, context):
    if not llm or not context or "No relevant information found" in context:
         return "N/A (No context retrieved)"
    if not answer or "Sorry, I encountered an issue" in answer or "An internal error occurred" in answer:
         return "N/A (Generation Error)"

    prompt = f"""
    You are an evaluator assessing whether an AI assistant's answer is faithful to the provided context.
    The answer must be fully supported by the context. It should not add external information or contradict the context.

    Context:
    ---
    {context}
    ---
    Question: {question}
    ---
    Answer: {answer}
    ---

    Based *only* on the provided context, is the Answer Faithful? Respond with only 'Yes' or 'No'.
    Faithful:
    """
    try:
        response = llm.generate_content(prompt)
        # Basic check for Yes/No - refine as needed
        evaluation = response.candidates[0].content.parts[0].text.strip()
        if "yes" in evaluation.lower():
            return "Yes"
        elif "no" in evaluation.lower():
            return "No"
        else:
            return "Uncertain" # LLM didn't give clear Yes/No
    except Exception as e:
        print(f"Error during faithfulness evaluation: {e}")
        return "Error"

def evaluate_relevance(question, answer, context):
    # Relevance is harder for LLM-as-judge, as it needs to understand the question's intent
    # A simpler check: Does the answer *address* the question, even if faithfulness is low?
    if not llm:
        return "N/A"
    if not answer or "Sorry, I encountered an issue" in answer or "An internal error occurred" in answer:
        return "N/A (Generation Error)"
    # Simple check for common "I don't know" patterns (adjust based on your prompt's output)
    if "couldn't find information" in answer or "don't have information" in answer:
        # If context *was* retrieved, but LLM said "I don't know", it might be irrelevant or unfaithful.
        # If no context was retrieved, this is the *correct* relevant answer.
        if context and "No relevant information found" not in context:
            return "Potentially No (Said 'I cannot answer' despite context)"
        else:
             return "Yes (Appropriately stated inability to answer)"

    prompt = f"""
    You are an evaluator assessing whether an AI assistant's answer is relevant to the user's question.
    Ignore faithfulness for now. Just determine if the answer *attempts* to address the core subject of the question.

    Question: {question}
    ---
    Answer: {answer}
    ---

    Is the Answer Relevant to the Question's topic? Respond with only 'Yes' or 'No'.
    Relevant:
    """
    try:
        response = llm.generate_content(prompt)
        evaluation = response.candidates[0].content.parts[0].text.strip()
        if "yes" in evaluation.lower():
            return "Yes"
        elif "no" in evaluation.lower():
            return "No"
        else:
            return "Uncertain"
    except Exception as e:
        print(f"Error during relevance evaluation: {e}")
        return "Error"

# --- Main Evaluation Loop ---
def run_evaluation():
    if not all([llm, embedding_model, collection]):
        print("ERROR: Backend components not initialized. Cannot run evaluation.")
        return

    results = []
    print(f"\n--- Starting Evaluation ---")
    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\nProcessing Question {i+1}/{len(TEST_QUESTIONS)}: {question}")

        start_time = time.time()
        rag_result = get_rag_response(question) # Get {'answer': ..., 'context': ...}
        end_time = time.time()

        latency = end_time - start_time
        answer = rag_result['answer']
        context = rag_result['context']

        # --- Perform Evaluations ---
        print(f"  Answer: {answer}")
        print(f"  Latency: {latency:.2f} seconds")
        # print(f"  Context: {context[:300]}...") # Optionally print context here too

        # LLM-assisted evaluation (can be slow due to extra API calls)
        faithfulness_score = evaluate_faithfulness(question, answer, context)
        relevance_score = evaluate_relevance(question, answer, context)
        print(f"  Faithfulness (LLM): {faithfulness_score}")
        print(f"  Relevance (LLM): {relevance_score}")

        # --- Manual Evaluation Prompt (Alternative/Complementary) ---
        # print("\n--- Manual Evaluation ---")
        # manual_faithfulness = input("Is the answer Faithful to the context? (y/n/na): ").lower()
        # manual_relevance = input("Is the answer Relevant to the question? (y/n/na): ").lower()
        # manual_notes = input("Any notes?: ")
        # -----------------------------------------------------------

        results.append({
            "Question": question,
            "Answer": answer,
            "Context": context,
            "Latency": latency,
            "Faithfulness (LLM)": faithfulness_score,
            "Relevance (LLM)": relevance_score,
            # "Manual Faithfulness": manual_faithfulness, # Uncomment if using manual
            # "Manual Relevance": manual_relevance,      # Uncomment if using manual
            # "Manual Notes": manual_notes             # Uncomment if using manual
        })

        # Introduce a delay before the next API call
        time.sleep(API_CALL_DELAY)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()