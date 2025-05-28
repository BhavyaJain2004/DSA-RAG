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

| Component                 | Description                                                                                                                                                                                                                                                               |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `scripts/scraping/`       | Contains web scrapers and data parsing scripts responsible for extracting DSA purports from various online sources (like LeetCode, GFG) and books (converted to text format).                                                                                             |
| `scripts/indexing/`       | Processes the raw text data, converts it into high-dimensional vector embeddings using an embedding model, and stores them in a FAISS vector store for efficient semantic search.                                                                                            |
| `src/rag_pipeline.py`     | The core RAG logic. Manages query processing, retrieves top-k most relevant information chunks from the vector store, constructs a strict prompt, and sends it to the LLM for response generation.                                                                           |
| `src/api.py`              | Flask-based API endpoint (`/chat`) responsible for handling incoming user queries from the frontend, orchestrating the RAG pipeline, managing chat history, and delivering JSON responses. This also handles requests for visualization data and ASCII diagrams.             |
| `src/visualizer.py`       | Logic for generating animated data structure visualizations. It receives data from `src/api.py` and returns structured data for frontend rendering.                                                                                                                         |
| `src/ascii_diagram_maker.py` | Contains algorithms to convert input data/concepts into precise ASCII diagrams for various DSA elements, integrated with `src/api.py` for dynamic generation.                                                                                                                |
| `backend.py`              | The main Flask application entry point, setting up routes, CORS, and integrating the `src/api.py` functionality.                                                                                                                                                          |
| `frontend/` (Vercel)      | The user-facing chat interface, built with [Your Frontend Framework, e.g., React/Next.js/Vue.js], deployed on Vercel. It interacts with the backend API to send queries and display responses, animations, and diagrams.                                                   |

## 🚀 Technologies Used

* **Backend:** Python, Flask, Gunicorn
* **AI/ML:** LangChain, HuggingFace Transformers
* **Embeddings:** `BAAI/bge-small-en`
* **Vector Database:** FAISS
* **LLMs:** Meta Llama-3.3-70B-Instruct-Turbo, Mistral-7B-Instruct-v0.1 (via Together.ai & Cohere APIs)
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
    Ensure you have your `vectorstore/db_faiss` folder populated with your pre-indexed DSA data. If not, you'll need to run your indexing script (`scripts/indexing/vec_indexing.py`) after setting up your scraping data.

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
