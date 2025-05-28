from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv, find_dotenv

# Import your existing RAG and LLM logic
# Ensure rag_query.py is in the same directory or its path is correctly configured
from rag_query import TOGETHER_API_KEY, DB_FAISS_PATH, get_vectorstore_instance, get_dsa_agent

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
# Enable CORS for all origins for development.
# In production, restrict this to your frontend's domain.
CORS(app, origins=["https://dsa-rag.vercel.app/"]) 

# --- Initialize your DSA resources (similar to Streamlit's @st.cache_resource) ---
# This part runs when the Flask app starts.
dsa_agent_instance = None

def initialize_dsa_resources():
    global dsa_agent_instance
    if dsa_agent_instance is None:
        if TOGETHER_API_KEY is None:
            print("Error: TOGETHER_API_KEY environment variable is not set.")
            return None
        try:
            vectorstore_db = get_vectorstore_instance()
            if vectorstore_db is None:
                print(f"Failed to load DSA knowledge base from {DB_FAISS_PATH}.")
                return None
        except Exception as e:
            print(f"Initialization Error: Could not load the DSA knowledge base. Details: {e}")
            return None
        try:
            dsa_agent_instance = get_dsa_agent(vectorstore_db)
        except Exception as e:
            print(f"Initialization Error: Could not initialize the DSA agent. Details: {e}")
            return None
    return dsa_agent_instance

# Initialize when app starts
with app.app_context():
    initialize_dsa_resources()
    if dsa_agent_instance:
        print("DSA agent initialized successfully.")
    else:
        print("DSA agent initialization failed. Check your environment variables and FAISS path.")


@app.route('/')
def index():
    return "Backend is running. Access the frontend via index.html"

@app.route('/chat', methods=['POST'])
def chat():
    if not dsa_agent_instance:
        return jsonify({"output": "Backend not fully initialized. Please check server logs."}), 500

    data = request.json
    user_input = data.get('input')
    chat_history = data.get('chat_history', []) # Expects format [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]

    if not user_input:
        return jsonify({"output": "No input provided."}), 400

    print(f"Received chat request: Input='{user_input}', History length={len(chat_history)}")

    # Convert history to the format your agent expects (if different from frontend)
    # Langchain agents often expect a list of BaseMessage (HumanMessage, AIMessage)
    # You might need to adjust this based on your `get_dsa_agent` implementation.
    # For now, let's assume it can handle [{'role': 'user/assistant', 'content': '...'}] or convert it.
    
    # Example conversion if your agent expects HumanMessage/AIMessage:
    converted_history = []
    # (Assuming you have HumanMessage/AIMessage from langchain.schema import HumanMessage, AIMessage)
    # from langchain.schema import HumanMessage, AIMessage
    # for msg in chat_history:
    #     if msg['role'] == 'user':
    #         converted_history.append(HumanMessage(content=msg['content']))
    #     elif msg['role'] == 'assistant':
    #         converted_history.append(AIMessage(content=msg['content']))

    try:
        # Use your agent's invoke method
        response_dict = dsa_agent_instance.invoke({
            "input": user_input,
            "chat_history": chat_history # Use the history as is, or converted_history
        })
        llm_response = response_dict.get("output", "I could not generate a response.")
        return jsonify({"output": llm_response})
    except Exception as e:
        print(f"Error invoking agent: {e}")
        return jsonify({"output": f"An error occurred while processing your request: {e}"}), 500

if __name__ == '__main__':
    # You can change the port if 5000 is in use
   app.run(debug=True, host='0.0.0.0', port=5000)