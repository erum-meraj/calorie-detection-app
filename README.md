# calorie-detection-app

# Nutritional Value Retrieval App

This Flask-based application retrieves the nutritional values of Indian dishes using a combination of Langchain, Google Generative AI (Gemini model), and PDF document data. The app attempts to fetch information from a PDF document first, and if the information is not available, it falls back on general knowledge using a predefined prompt.

## Features

- **Conversational Retrieval**: Retrieves nutritional values based on queries using a combination of document context (PDF) and fallback on a general knowledge LLM model.
- **Chroma Vector Store**: Efficiently stores and retrieves document embeddings for fast lookup.
- **PDF Integration**: Uses Langchain's `PyPDFLoader` to load and chunk a PDF containing nutritional data.
- **Google Generative AI**: Leverages Google's generative AI model for intelligent fallback in cases where the document lacks necessary information.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Replace the `GOOGLE_API_KEY` in the code with your actual API key from Google.

4. **Prepare PDF document:**
   Ensure the nutritional values PDF document (`nutri_value.pdf`) is located in the same directory as the script.

## Usage

1. **Run the Flask App:**

   ```bash
   python app.py
   ```

2. **Make POST requests:**
   Send a `POST` request to the `/predict` endpoint with the following format:

   ```json
   {
     "dish": "Paneer Tikka",
     "size": "2 servings"
   }
   ```

3. **Response Format:**
   The API returns a JSON object with the nutritional breakdown, including the amount and daily value of various nutrients, e.g.:

   ```json
   {
     "calories": {
       "amount": "300 kcal",
       "daily_value": "15%"
     },
     "protein": {
       "amount": "10g",
       "daily_value": "20%"
     }
   }
   ```

## Workflow

1. **Document Retrieval**: The app first tries to retrieve the nutritional data from the provided PDF by using a Chroma vector store to match the user's query.
2. **Fallback Mechanism**: If the document doesn't contain the relevant information, the app uses a predefined prompt with the Google Generative AI model to generate a response.
3. **Response Parsing**: The app parses the response to ensure it adheres to the expected format (key-value pairs with amounts and daily values).

## API Endpoints

- `POST /predict`: Retrieves the nutritional values for a given dish and serving size. Requires the following parameters:
  - `dish`: Name of the dish (e.g., "Paneer Tikka").
  - `size`: Size/portion (e.g., "2 servings").

## Error Handling

- **Invalid Input**: If `dish` or `size` is missing, the API will return an error with status code `400`.
- **Fallback on Failure**: If any error occurs in the retrieval chain or the PDF does not contain the information, the model falls back to general knowledge retrieval.

## Persistence

The application stores the document embeddings in the `chroma_db` directory, which persists between sessions for faster retrieval.

## Requirements

- Flask
- Flask-CORS
- Langchain
- Google Generative AI (Gemini)
- PyPDFLoader
- Chroma

## License

This project is licensed under the MIT License.
