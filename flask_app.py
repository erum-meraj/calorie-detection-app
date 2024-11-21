from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# Replace with your Google API Key
GOOGLE_API_KEY = "AIzaSyAVOmym8rZ0YLbCaXVLDxytC4RK82ezk-s"

# Initialize the Gemini model (using ChatGoogleGenerativeAI with a prompt)
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5,
    convert_system_message_to_human=True
)

# Define a default prompt template for fallback general knowledge
default_prompt_text = """You have the entire knowledge about the nutritional values regarding any of the Indian dishes. I will give you the name of the dish and the amount in the form of number of servings, and you have to give me the nutritional values in the form of output shown below. For each of the nutrients in the output, you have to tell me the amount and the daily value(%). MAKE SURE THE VALUES YOU GIVE ARE CORRECT UP TO YOUR KNOWLEDGE.

The input should be like:
Dish: {dish}
Size: {size}

The output should be like:

Calories: AMOUNT (DAILY VALUE)
Protein: AMOUNT (DAILY VALUE)
Carbohydrates: AMOUNT (DAILY VALUE)
Fats: AMOUNT (DAILY VALUE)
Fiber: AMOUNT (DAILY VALUE)
Sugars: AMOUNT (DAILY VALUE)
Cholesterol: AMOUNT (DAILY VALUE)
Sodium: AMOUNT (DAILY VALUE)
Iron: AMOUNT (DAILY VALUE)
Calcium: AMOUNT (DAILY VALUE)

AVOID ANY OTHER TEXT IN THE OUTPUT AND MAKE SURE THAT THE OUTPUT IS IN THE FORMAT SHOWN ABOVE. THERE MUST BE PARENTHESES IN THE DAILY VALUE FOR EACH OF THE NUTRIENTS."""
default_prompt = PromptTemplate(template=default_prompt_text, input_variables=["dish", "size"])

# Load PDF document and split into chunks
pdf_path = os.path.join(os.path.dirname(__file__), "nutri_value.pdf")
pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load_and_split()  # Split the PDF into pages/chunks
texts = [page.page_content for page in pages]  # Extract content from each page

# Create embeddings using the Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Define the persistence directory
PERSIST_DIRECTORY = "chroma_db"

# Check if the persistence directory exists to decide whether to load or create
if os.path.exists(PERSIST_DIRECTORY):
    # Load existing Chroma vector store
    chroma_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
else:
    # Create a new Chroma vector store and add texts
    chroma_store = Chroma.from_texts(texts, embeddings, persist_directory=PERSIST_DIRECTORY)

# Persist the vector store to disk
chroma_store.persist()

# Create a retriever from the Chroma vector store
vector_retriever = chroma_store.as_retriever(search_kwargs={"k": 5})

# Set up a ConversationalRetrievalChain with the LLM and retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vector_retriever
)

def parse_response_to_json(response):
    # Split the response into lines
    lines = response.strip().split('\n')
    
    # Create a dictionary to store the parsed data
    data = {}
    
    # Regular expression to match the format "Key: Amount (Daily Value)"
    pattern = r'(.+):\s*(.+)\s*\((.+)\)'
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            key, amount, daily_value = match.groups()
            key = key.strip().lower().replace(' ', '_')
            data[key] = {
                "amount": amount.strip(),
                "daily_value": daily_value.strip()
            }
    
    return data

# Define a function to handle user queries and pass them through the RAG pipeline
def res(dish, size):
    # Format the prompt for the LLM
    prompt = f"""Dish: {dish}
Size: {size}

Provide the nutritional value for this dish and size. If the information is not available, respond with 'NOT_FOUND'."""
    
    try:
        # Attempt to retrieve information from the PDF
        result = qa_chain({
            "question": prompt,
            "chat_history": []  # Provide an empty chat history for independent calls
        })
        
        # Check if the response indicates no useful information was found
        if "NOT_FOUND" in result['answer'] or "does not contain" in result['answer']:
            # Fallback to general model knowledge with the default prompt
            fallback_prompt = default_prompt.format(dish=dish, size=size)
            general_knowledge_response = model.invoke(fallback_prompt)
            return parse_response_to_json(general_knowledge_response.content)

        return parse_response_to_json(result['answer'])
    
    except Exception as e:
        # If any error occurs (e.g., model API failure), return the fallback response
        fallback_prompt = default_prompt.format(dish=dish, size=size)
        general_knowledge_response = model.invoke(fallback_prompt)
        return parse_response_to_json(general_knowledge_response.content)

# API route for making requests with dish and size
@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    
    if not request_data or "dish" not in request_data or "size" not in request_data:
        return jsonify({"error": "Invalid request, 'dish' and 'size' are required"}), 400
    
    dish = request_data.get("dish")
    size = request_data.get("size")
    
    # Retrieve the response using the RAG pipeline (LLM + PDF context)
    response = res(dish, size)
    print(response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
