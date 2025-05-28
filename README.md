# 📚 Algorithm Architect: Your AI-Powered DSA Learning Companion 🚀

This is **Algorithm Architect**, a comprehensive Retrieval-Augmented Generation (RAG) based AI solution designed to revolutionize Data Structures & Algorithms (DSA) learning and problem-solving. It's built to answer your DSA questions accurately, provide visual insights, and generate ready-to-use code, making complex concepts easy to grasp.

## ✨ Project Overview

Algorithm Architect serves as your personal DSA AI tutor, capable of understanding intricate queries and delivering precise, context-rich answers. It goes beyond simple Q&A by integrating powerful visualization and code generation capabilities.

### 🧠 What it does

* **Intelligent RAG Pipeline:** Leverages a sophisticated RAG architecture to fetch highly relevant information from a vast DSA knowledge base.
* **Contextual Q&A:** Answers Data Structures and Algorithms questions accurately and comprehensively based on retrieved context, minimizing hallucinations.
* **Multi-Language Code Generation:** Generates executable code solutions in any programming language (e.g., Python, C++, Java) for DSA problems.
* **Detailed Output Explanations:** Provides clear and concise explanations for the generated code and answers, ensuring concept clarity.
* **Inbuilt Visualizer:** Offers animated visualizations of core data structures like **Arrays, Stacks, Queues, and Linked Lists**, helping users understand their operations dynamically.
* **ASCII Diagram Maker:** Generates intuitive ASCII diagrams for various Data Structures and Algorithms based on user input, aiding in conceptual understanding.

### 📖 Data Sources

Our comprehensive knowledge base is meticulously curated from renowned resources to ensure accuracy and depth:

* **Books:**
    * "Data Structures and Algorithms in Python"
    * "Grokking Algorithms"
    * "Introduction to Algorithms - 3rd Edition"
* **Websites:**
    * LeetCode
    * GeeksforGeeks (GFG)

## 🛠️ How it Works

The "Algorithm Architect" operates through a robust backend pipeline that integrates advanced AI models with a meticulously structured knowledge base.

| Component Path                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/`                             | Stores the raw source data for the knowledge base, including PDF files of famous DSA books and JSON files containing scraped data from websites like LeetCode and GFG.                                                                                                                                                                                                                                                                                                                                                            |
| `scripts/scraping/`                 | Contains the Python code files dedicated to scraping data from various websites (e.g., LeetCode, GFG) and extracting content from PDF book files, preparing it for further processing.                                                                                                                                                                                                                                                                                                                                                |
| `simplerag.py`                      | This is the core data processing and indexing script. It takes the raw data from `data/`, converts the books and fetched data into manageable chunks, generates vector embeddings for these chunks, and then stores them efficiently into the FAISS vector store.                                                                                                                                                                                                                                                                |
| `vectorstore/`                      | This directory stores the generated embeddings and the FAISS (Facebook AI Similarity Search) index. This vector store is crucial for rapidly retrieving contextually relevant information during the RAG process.                                                                                                                                                                                                                                                                                                            |
| `rag_query.py`                      | Contains the core RAG logic for query processing. This file is responsible for creating and managing QA chains, setting up agents, defining custom prompts, and integrating various tools (like code generation) to facilitate intelligent and accurate responses.                                                                                                                                                                                                                                                            |
| `backend.py`                        | The main Flask application file. It sets up the web server, defines API routes (e.g., `/chat`), handles incoming requests from the frontend, orchestrates the RAG pipeline by calling `rag_query.py`, and manages the overall backend operations.                                                                                                                                                                                                                                                                               |
| `app/static/algo_visualizer_template.html` | Contains the HTML, CSS, and JavaScript code responsible for the interactive Data Structures & Algorithms visualizer. This component takes structured data from the backend to display animations of Arrays, Stacks, Queues, and Linked Lists.                                                                                                                                                                                                                                                                            |
| `index.html`                        | The primary HTML file for the frontend application. This is the main entry point for the user interface, serving as the chat interface where users interact with the AI, view visualizations, and receive responses.                                                                                                                                                                                                                                                                                                               |
| `frontend/` (Vercel)                | The complete user-facing chat interface, likely built with a modern JavaScript framework (e.g., React/Next.js/Vue.js), deployed on Vercel. It interacts with the `backend.py` API to send user queries and display responses, visualizations, and ASCII diagrams. (Note: `index.html` is part of this frontend.)                                                                                                                                                                                                                          |

## 🚀 Technologies Used

* **Backend:** Python, Flask, Gunicorn
* **AI/ML:** LangChain, HuggingFace Transformers
* **Embeddings:** `BAAI/bge-small-en`
* **Vector Database:** FAISS
* **LLMs:** Meta Llama-3.3-70B-Instruct-v0.1, Mistral-7B-Instruct-v0.1 (via Together.ai & Cohere APIs)
* **Deployment:** DigitalOcean Droplets (Backend), Vercel (Frontend)
* **Data Sourcing:** Custom Web Scrapers, Text Processing

## 🏃‍♀️ Running Locally

To get Algorithm Architect running on your local machine for development or testing:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BhavyaJain2004/DSA-RAG.git](https://github.com/BhavyaJain2004/DSA-RAG.git)
    cd DSA-RAG
    ```

2.  **Set up a Python Virtual Environment:**
    ```bash
    python3 -m venv venv_dsa_backend
    source venv_dsa_backend/bin/activate  # On Windows use: .\venv_dsa_backend\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    ```

4.  **Set up environment variables:**
    In the root directory of the project, create a file named `.env`. This file will store the environment variables needed by the app.
    Add the following lines to it, replacing placeholders with your actual API keys:
    ```dotenv
    TOGETHER_API_KEY=your_together_ai_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    HF_TOKEN=your_huggingface_token_if_needed # Required for HuggingFace models/datasets
    ```

5.  **Prepare your FAISS Vector Store:**
    Ensure you have your `vectorstore/db_faiss` folder populated with your pre-indexed DSA data. If not, you'll need to run your indexing script (`simplerag.py`) after setting up your `data/` folder.
    ```bash
    python simplerag.py
    ```

6.  **Run the Backend App:**
    Once everything is set up, run the backend server using Gunicorn:
    ```bash
    gunicorn backend:app --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 120
    ```
    After it starts, your backend API will be accessible at `http://localhost:5000`.

## 🌐 Live Demo

Experience Algorithm Architect live! Interact with the AI, visualize DSA concepts, and generate code:

**Live Frontend Demo:** [**https://your-vercel-frontend-domain.vercel.app/**](https://your-vercel-frontend-domain.vercel.app/) (Replace with your actual Vercel deployment URL)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the [Your License Here, e.g., MIT License](LICENSE).

---
