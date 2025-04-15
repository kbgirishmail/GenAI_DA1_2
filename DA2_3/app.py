import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

# --- Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Gemini API key
# For better security, use environment variables: os.getenv("GEMINI_API_KEY")
API_KEY = "*************************************"
# FAQ_FILE = "faq.csv"
FAQ_FILE = "Maktek_faq.csv"
DB_DIRECTORY = "chroma_db" # Directory to store ChromaDB data
COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Good default, runs locally

# --- Initialize Gemini ---
try:
    genai.configure(api_key=API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'
    print("Gemini AI Model Initialized Successfully.")
except Exception as e:
    print(f"Error initializing Gemini AI: {e}")
    llm = None # Set llm to None if initialization fails

# --- Initialize Embedding Model ---
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Sentence Transformer Model Initialized Successfully.")
except Exception as e:
    print(f"Error initializing Sentence Transformer: {e}")
    embedding_model = None

# --- Initialize ChromaDB ---
try:
    # Using persistent storage
    client = chromadb.PersistentClient(path=DB_DIRECTORY)

    # Try to get the collection, or create it if it doesn't exist
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Existing ChromaDB collection '{COLLECTION_NAME}' loaded.")
        # Optional: Check if it needs updating based on FAQ file modification time
    except:
        print(f"Creating new ChromaDB collection '{COLLECTION_NAME}'.")
        collection = client.create_collection(name=COLLECTION_NAME)

        # --- Load and Index Data (Only if collection is new or needs update) ---
        if embedding_model:
            print("Loading data from CSV...")
            df = pd.read_csv(FAQ_FILE)
            # Ensure columns exist
            if 'Question' not in df.columns or 'Answer' not in df.columns:
                raise ValueError("CSV must contain 'Question' and 'Answer' columns")

            print(f"Found {len(df)} FAQ entries.")
            documents = []
            metadatas = []
            ids = []

            print("Generating embeddings and preparing data for indexing...")
            for index, row in df.iterrows():
                # We store the Answer as the document, and Question in metadata
                # You could also combine Q&A for embedding if desired
                answer_text = str(row['Answer'])
                question_text = str(row['Question'])
                documents.append(answer_text)
                metadatas.append({'question': question_text})
                ids.append(f"faq_{index}") # Simple unique ID

            if documents:
                # Generate embeddings in batches (more efficient for large datasets)
                embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
                print(f"Generated {len(embeddings)} embeddings.")

                # Add to ChromaDB collection
                collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print("Data successfully added to ChromaDB collection.")
            else:
                 print("No documents found to add to the collection.")
        else:
             print("Skipping data indexing due to embedding model initialization failure.")

except Exception as e:
    print(f"Error initializing ChromaDB or indexing data: {e}")
    collection = None # Set collection to None if initialization fails

# --- Initialize Flask App ---
app = Flask(__name__)

# --- RAG Function ---
def get_rag_response(query, n_results=3):
    if not all([llm, embedding_model, collection]):
        return "Error: Backend components not initialized."

    try:
        # 1. Embed the user query
        query_embedding = embedding_model.encode([query]).tolist()

        # 2. Query ChromaDB for relevant documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results, # Number of relevant documents to retrieve
            include=['documents', 'metadatas'] # Include documents and metadata
        )

        # 3. Extract context
        # Check if results and documents are present and not empty
        if not results or not results.get('documents') or not results['documents'][0]:
            print("No relevant documents found in ChromaDB.")
            # Optional: Return a specific message or try LLM without context
            # For this basic example, we'll let the LLM handle it based on prompt
            context = "No relevant information found in the FAQ database."
        else:
             # Join the retrieved document snippets (answers) into a single context string
            context = "\n---\n".join(results['documents'][0])
            print(f"Retrieved Context:\n{context[:500]}...") # Print first 500 chars of context

        # 4. Build the prompt for the LLM
        prompt_template = f"""
        You are a helpful AI assistant answering questions based ONLY on the provided context from our company's FAQ.
        Your goal is to provide a concise and accurate answer derived *directly* from the text snippets below.
        Do not use any prior knowledge or information outside of the provided context.
        If the context does not contain the information needed to answer the question, state that clearly.

        Context from FAQ:
        ---
        {context}
        ---

        User Question: {query}

        Based *only* on the context above, please provide the answer:
        """
        # print(f"\nPrompt being sent to LLM:\n{prompt_template}\n") # Uncomment for debugging

        # 5. Call the LLM API
        try:
            response = llm.generate_content(prompt_template)
            # Accessing the text safely
            if response and response.candidates and response.candidates[0].content.parts:
                 llm_answer = response.candidates[0].content.parts[0].text
            else:
                # Handle cases where the response structure is unexpected
                print("Warning: Unexpected LLM response structure:", response)
                llm_answer = "Sorry, I encountered an issue generating the response."

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            llm_answer = f"Sorry, there was an error contacting the AI service: {e}"

        # return llm_answer
        return {
            'answer': llm_answer,
            'context': context
        }

    except Exception as e:
        print(f"Error during RAG process: {e}")
        return f"An internal error occurred: {e}"


# --- Flask API Endpoint ---
@app.route('/')
def home():
    # Serve the HTML interface
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    print(f"\nReceived query: {query}")
    answer = get_rag_response(query)['answer']
    print(f"Generated answer: {answer}")

    return jsonify({"answer": answer})

# --- Main execution ---
if __name__ == '__main__':
    if not all([llm, embedding_model, collection]):
         print("\nERROR: One or more components (LLM, Embedding Model, ChromaDB) failed to initialize. Exiting.")
    else:
        print("\nAll components initialized. Starting Flask server...")
        # Note: use_reloader=False can help prevent re-running the indexing on save
        # During development, you might want it True, but be mindful of re-indexing time.
        app.run(debug=True, use_reloader=False, port=5000)

