from dotenv import load_dotenv , find_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings 


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent,Tool,AgentType
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.retrievers import ContextualCompressionRetriever
from flashrank import Ranker
import uuid 

from langchain.retrievers.document_compressors import CohereRerank
from langchain_cohere import CohereRerank
from langchain_community.llms import Together
import requests
import base64
from together import Together as TogetherClient

import warnings # <--- Add this line!

# Suppress all warnings
warnings.filterwarnings('ignore') 
load_dotenv(find_dotenv())


HF_TOKEN = os.environ.get("HF_TOKEN")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")



# TOGETHER_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
TOGETHER_MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore_instance():
   
    try:
        embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en")
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading FAISS vector store: {e}") # Use print, not st.error
        return None


if not COHERE_API_KEY:  
    print("Error: COHERE_API_KEY environment variable not set. Please set it in your .env file.")
    exit()

def load_llm(together_model_id,temperature:float=0.0):

    llm = Together(
        model = together_model_id,
        temperature = temperature,
        max_tokens = 512,
        together_api_key = TOGETHER_API_KEY
    )
    return llm
def get_dsa_agent(db_instance):
        if db_instance is None:
            raise ValueError("Database instance cannot be None when initializing agent.")

        base_retriever = db_instance.as_retriever(search_kwargs={"k":25})
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )


        strict_qa_prompt = PromptTemplate(
            template="""You are an AI assistant specialized in Data Structures and Algorithms (DSA).
            Use the following pieces of context from DSA-related documents to answer the question at the end.
            If the answer cannot be found *explicitly* within the provided context, please state: "I cannot answer this question based on the provided DSA knowledge base."
            Do NOT use your internal knowledge or general information. Stick strictly to the provided context.
            If a programming language is specified in the question (e.g., "Java", "Python"), provide code snippets ONLY in that language.
            When providing code snippets, ensure they are complete, runnable, and presented within a markdown code block (```language\ncode\n```) without any additional conversational text outside the block.

            Context:
            {context}
            Question: 
            {question}

            Helpful Answer:""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm = load_llm(TOGETHER_MODEL_ID),
            chain_type = "stuff",
            retriever = compression_retriever,
            return_source_documents = True,
            chain_type_kwargs = {"prompt":strict_qa_prompt}
        )
        print("RAG chain with re-ranking (compression_retriever) ready.")


        def get_knowledge_from_rag(query:str) -> str:
            """
            Useful for answering questions about programming concepts, algorithms,
            data structures, and retrieving relevant code snippets from stored books and JSON documents.
            ALWAYS use this tool first for queries seeking specific information/theory or code examples
            that should be within your learned knowledge (from your books/JSONs).
            """

            response = qa_chain.invoke({"query":query})
            print("\n--- RAG Source Documents (Retrieved) ---")

            if "source_documents" in response and response["source_documents"]:
                # Print details for the first 2 documents
                for i, doc in enumerate(response["source_documents"][:2]): # Get only top 2 docs
                    source_info = doc.metadata.get('source', 'N/A')
                    print(f"Doc {i+1} (Source: {source_info}): {doc.page_content[:150]}...") # Print first 150 chars
            else:

                print("No relevant source documents found for this query in the knowledge base.")
            print("--- End Source Documents ---\n")

            return response["result"]

        rag_tool = Tool(
                name = "Knowledge Base",
                func=get_knowledge_from_rag,
                description="""**ONLY** useful for answering questions *strictly* about Data Structures and Algorithms (DSA) concepts, specific programming topics, algorithms, data structures, and **retrieving EXISTING code snippets from the loaded DSA documents**.
            **This tool MUST be prioritized for any request involving existing code examples or theoretical DSA information.**
            **If the user's question is NOT explicitly and directly about DSA or programming (e.g., general knowledge, current events, factual information outside of code/algorithms), then DO NOT use this tool.**
            Instead, if the question is out of the DSA domain, the agent should state: 'I am specialized in Data Structures and Algorithms and can only answer questions related to that domain. I cannot answer general knowledge questions.'
            """
            )   
        print("Tool: 'Knowledge Base' (your RAG system) defined.")


        def generate_code(request:str) ->str:
            """
            Generates programming code based on a user's request.
            Use this tool ONLY when the user explicitly asks for code generation for a NEW problem
            and the 'Knowledge Base' tool did not provide a satisfactory or relevant existing code snippet.
            Input should be a clear and concise description of the desired code, specifying the language if needed.
            Example: 'Java function to reverse a string', 'Python class for a binary tree'.
            """

            prompt_string =  f"""You are an expert programmer. Your task is to generate clean, correct, and runnable code based on the user's request.
            If the user's request is not directly about generating code, state 'I cannot generate code for that specific request.'.
            ONLY provide the code in a markdown code block, do not add any explanations or extra text outside the code block.
            If the user specifies a language, generate code in that language.

            User request = {request}
            Code:
            ```
            """
            response = load_llm(TOGETHER_MODEL_ID).invoke(prompt_string, stop=["```", "User request ="]) 
            if not response.endswith("```"):
                response += "```" # Add closing backtick if it was cut off by stop sequence

            
            generated_code_with_source = f"{response}\n\n**Source: AI Generated Code**"
            return generated_code_with_source


        code_generator_tool = Tool(
            name = "Code Generator",
            func=generate_code,
            description="""Generates **NEW** programming code snippets based on a user's **explicit request for creation**.
            **This tool should ONLY be used as a LAST RESORT if the 'Knowledge Base' tool fails to provide a suitable existing code example or if the user explicitly asks for new code generation (e.g., "write a function for...", "create a class...").**
            Specify the language if you need code in a language other than Python.
            """
        )
        print("Tool: 'Code Generator' defined using direct LLM call (with internal prompt string).")


        python_repl_tool = PythonREPLTool(
            name = "Python REPL",
            description="""Executes Python code. Input: single string of Python code. **Use `print()` for output.**
            Output: code execution result.
            Use for testing, debugging, and demonstrating output.
            Do NOT generate code or perform unrelated math.
            """
        )
        print("Tool: 'Python REPL' defined.")



        def generate_ascii_dsa_diagram(concept_description: str) -> str:

            """
            Generates a clear ASCII art diagram for a given Data Structure or Algorithm concept.
            The input should be a precise description of the desired diagram.
            """

            prompt_string = f"""You are an expert at generating clear, minimalist, and easy-to-understand ASCII art diagrams for Data Structures and Algorithms concepts.
            Generate ONLY the ASCII art diagram based on the user's request.
            Crucially, ensure the ASCII diagram is enclosed in a **single markdown code block** (e.g., ````\n[ASCII here]\n````).
            Do NOT add `python` or any other language specifier to the code block for ASCII diagrams.
            The opening ```` must be on a separate line, and the closing ```` must be on a separate line.
            DO NOT add any extra text, explanations, or conversational filler outside the ASCII art block.
            The diagram should be simple and use standard ASCII characters.

            User request for diagram: {concept_description}

            ASCII Diagram:
            ```
            """
            try:
                raw_ascii_art = load_llm(
                    TOGETHER_MODEL_ID,
                    temperature=0.0 # Low temperature for consistent output
                ).invoke(
                    prompt_string,
                    stop=["```\n", "ASCII Diagram:", "Thought:", "Action:", "\nObservation:", "Final Answer:",
                        "\nFor troubleshooting, visit:", "\nAnswer:", "\n- ", "\nNote:",
                        "\nFor further assistance:", "\nAdditional Info:", "\nDisclaimer:",
                        # Removed specific ```python stop, relying on generic ```
                        "```", # Keep this generic ``` to catch closing blocks
                        ]
                )
                
                cleaned_ascii_art = raw_ascii_art.strip()
                if cleaned_ascii_art.startswith("```"):
                    cleaned_ascii_art = cleaned_ascii_art[3:]
                if cleaned_ascii_art.endswith("```"):
                    cleaned_ascii_art = cleaned_ascii_art[:-3]

            
                return f"```\n{cleaned_ascii_art}\n```"

            except Exception as e:
                return f"Error generating ASCII diagram: {e}"

        ascii_visualizer_tool = Tool(
            name="ASCII Visualizer",
            func=generate_ascii_dsa_diagram,
            description="""Useful for generating simple, text-based (ASCII art) diagrams of Data Structures and Algorithms.
            Use this when a quick, clear, character-based visual representation is beneficial for concepts like arrays, linked lists, simple trees, queues, stacks, or graph traversals.
            **When comparing two concepts, generate a clean, line-based ASCII table comparison (like with a '|' separator) to clearly highlight their differences.**
            **Use this tool not only when the user explicitly asks for a diagram or visual, but also when you believe a visual representation would significantly enhance the user's understanding of the concept being discussed, even if not explicitly requested.**
            The input should be a precise and clear description of the diagram needed, focusing on clarity, minimalism, and the specific structure to be visualized.
            Example Input: "ASCII art of a horizontal array with 5 elements labeled with indices", "ASCII diagram of a singly linked list with nodes A, B, C.", "ASCII table comparing Stack and Queue."
            """
        )

        tools = [
            rag_tool,
            code_generator_tool,
            python_repl_tool,
            ascii_visualizer_tool

        ]
            
        agent = initialize_agent(
            tools=tools,
            llm = load_llm(TOGETHER_MODEL_ID),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose = True,
            handle_parsing_errors = True,
            agent_kwargs={
            "prefix": """You are an expert Data Structures and Algorithms (DSA) tutor and problem-solver.
            You have access to the following tools:""",
            "suffix": """Begin!

            {chat_history}
            Question: {input}
            {agent_scratchpad}""",
            "format_instructions": """To respond to the user, you must follow this precise format:

            Thought: You must always first think step-by-step about what to do, considering the question and your available tools.
            Action: The tool to use, must be one of [{tool_names}]
            Action Input: The input to the action, in a clear and concise string.
            Observation: The result of the action

            ... (This Thought/Action/Action Input/Observation sequence can repeat multiple times if necessary.
            Ensure that when you use a tool, you provide the `Action Input` in the exact format required by the tool's description.) ...

            Thought: Now that I have all the information, I can provide a final answer.

            
            Final Answer: The complete, direct answer to the user's request. This must be the final response and should not contain any further actions or thoughts. [Your final answer should be in markdown format. If you have code, put it in a ```language``` block. If you have ASCII art, put it in a ```text``` or ```code``` block. Present all information clearly.]
            **CRUCIAL: The Final Answer MUST NOT contain any tool actions, thoughts, or any markdown code blocks after the "Final Answer:" keyword itself.**
            **If the answer includes a code block or ASCII art that was already formatted by a tool (i.e., it already contains triple backticks ```), present that content exactly as received, DO NOT WRAP IT IN NEW MARKDOWN BACKTICKS (```) OR ANY OTHER CONVERSATIONAL FILLER DIRECTLY AROUND IT within the Final Answer.**
            **IMPORTANT: ABSOLUTELY AVOID ANY REPETITION. Do NOT repeat any code blocks, ASCII diagrams, or textual explanations that have already been presented in full. Only include each distinct piece of information once. Be concise and directly answer the user's query.Do not give more than 1 example for the same concpept or by changing numerical values.**
            **Do NOT include any extra notes, disclaimers, or troubleshooting information (like "For troubleshooting, visit:") after the final answer. The Final Answer should be ONLY the answer itself.**


            **Tool Selection Guidelines:**

            1.  **Knowledge Base:**
                * **PRIORITY 1: ABSOLUTELY USE THIS for all questions seeking existing DSA concepts, definitions, algorithms, data structures, comparisons (e.g., "compare X and Y"), or existing code snippets from your loaded DSA documents.**
                * **Example Use:** "What is a Binary Search Tree?", "Explain QuickSort algorithm.", "Show Python code for Dijkstra's algorithm from the knowledge base.", "Compare Dijkstra's and Bellman-Ford."

            2.  **Code Generator:**
                * **PRIORITY 2 (If Knowledge Base fails):** Use this **ONLY** when the user explicitly requests to **create NEW code** for a problem that is not directly found in the knowledge base, or if the user wants code for a *novel* problem.
                * **Example Use:** "Write a Java function to implement a custom queue with two stacks.", "Generate Python code for a simple hash map."

            3.  **Python REPL:**
                * **Use this tool to execute Python code snippets for testing, debugging, or demonstrating the output of code. **This tool can run any valid Python code, including user-provided; prioritize user code execution. **
                * **Important:** If using for a calculation or testing, **always use `print()`** to show the output clearly. Do not perform complex mathematical calculations not related to coding. **DO NOT use this tool for explanations, comparisons, or generating code.**
                * When displaying results, show the input code, then its output clearly.

            4.  **ASCII Visualizer:**
                ** Use this tool to create an ASCII art diagram or visual representation of a data structure (like arrays, linked lists, stacks, queues, trees, graphs) or an algorithmic process. **Use this tool not only when the user explicitly asks for a diagram or visual, but also when you believe a visual representation would significantly enhance the user's understanding of the concept being discussed, even if not explicitly requested.** Provide a clear and concise description of what you want to visualize in the 'Action Input'.
                * **Input:** The `Action Input` must be a very precise and concise description of the ASCII diagram needed.
                * **Output Format Rule (VERY IMPORTANT FOR ASCII DIAGRAMS):** **The ASCII diagram MUST be returned inside a simple markdown code block (` ```\n[ASCII here]\n``` `). Do NOT include `python` or any other language specifier for ASCII art.**
                * **Example Action Input:** "ASCII art of a horizontal array with 5 elements labeled with indices", "ASCII diagram of a singly linked list with nodes A, B, C.", "ASCII table comparing Stack and Queue."


            **Important Constraints:**

            * If the user's question is **NOT** related to Data Structures and Algorithms or programming concepts, state: "I am specialized in Data Structures and Algorithms and can only answer questions related to that domain."
            * **Do NOT** include any 'Thought', 'Action', 'Action Input', or 'Observation' steps after a 'Final Answer:'. A 'Final Answer:' terminates the interaction.
            """,
            "input_variables": ["input", "chat_history", "agent_scratchpad"],
                "stop": ["\nObservation:", "\nFinal Answer:",
                    "\nFor troubleshooting, visit:", "\nAnswer:",
                    "\n- ", "\nNote:", "\nFor further assistance:", "\nAdditional Info:",
                    "\nDisclaimer:", "```\n\n```", "\nAction: None", # Keep these
                    "\nThought:", # <--- ADD THIS! This is crucial to stop the internal loop
                    "\nAction:",   # <--- ADD THIS! Also important to stop action attempts after FA
                    "```python", "```java", "```javascript", "```", # Catch any code block opening
                    "\nI hope this helps", "\nLet me know", "\nBest regards", # Catch common conversational fillers
                    "\nI cannot answer this question based on the provided DSA knowledge base." # Catch full refusal phrase
                    ]
            }
        )
        return agent


if __name__ == "__main__":
    print("\nInitializing resources for direct execution...")
    # For direct execution, we load the vectorstore here
    direct_db_instance = get_vectorstore_instance() # Call the new function
    if direct_db_instance:
        # Create the agent for direct execution
        direct_agent = get_dsa_agent(direct_db_instance)
        print("Agent initialized successfully for direct execution.")
        print("Ready to take queries!")
        query = input("Write a Query : ")
        direct_agent.invoke({"input": query, "chat_history": []})
    else:
        print("Failed to initialize agent for direct execution due to vector store error.")