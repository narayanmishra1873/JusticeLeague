import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer  # Use CountVectorizer instead of FastEmbed
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')

from langchain_mistralai import ChatMistralAI

# Initialize Flask app
app = Flask(__name__)

# Initialize the Chat model
model = ChatMistralAI(model="open-mistral-nemo", MISTRAL_API_KEY=api_key)

# Load embeddings and documents
current_dir = os.getcwd()
embeddings_path = os.path.join(current_dir, "embeddings_ipc.pkl")

# Load embeddings and documents
with open(embeddings_path, 'rb') as f:
    embeddings, docs = pickle.load(f)

# Initialize CountVectorizer for document embeddings
vectorizer = CountVectorizer()
embeddings = vectorizer.fit_transform([doc.page_content for doc in docs]).toarray()

# Query embeddings function
def query_embeddings(query_text):
    # Transform the query into the same vector space as the embeddings
    query_embedding = vectorizer.transform([query_text]).toarray()
    # Calculate distances using Euclidean distance
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    return distances

@app.route('/get_advice', methods=['POST'])
def get_advice():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    distances = query_embeddings(query)
    k = 3
    closest_indices = np.argsort(distances)[:k]
    relevant_docs = [docs[i] for i in closest_indices]

    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide an answer based only on the provided documents."
        + "\n\nYour response should not be rude. Don't let the user know which document you are referring to."
        + "\n\nIf you don't know the answer then refrain from answering anything!"
    )

    messages = [
        SystemMessage(content="""You are a Lawyer. You provide the best legal advice. In general, There are three scenerios based on the question the user asks:
                                A) The user greets with 'Hi' or 'hello' or the user is asking a general question which is not related to legal issues or the user is having a normal question:
                                    You do the following
                                    1) Answer the question asked by the user.
                                    2) Tell him that you are a legal chatbot assistant and would be happy to help with any legal queries they have.
                                B) The user is asking a genuine legal question and the user is the victim and has done nothing wrong.:
                                    You tell him the following 
                                    1) The relevant IPC section codes and describe them using the context given by the user.
                                    2) His Rights
                                    3) What evidence should they collect. 
                                    4) What he should do
                                    5) What he should avoid to do
                                C) The user is the culprit or has done something wrong:
                                    Start by telling him that he is at wrong and shouldnt have done the deed in the first place and then:
                                    1) The relevant IPC sections that the user can be charged against by the victim and its description based on the given context.
                                    2) The victim's rights
                                    3) The users rights
                                    4) What evidence should they collect to somehow prove he is innocent. 
                                    5) What he should do
                                    6) What he should avoid to do
                            Dont tell the user about the above categories.
                            You the answer in the easiest way to understand."""), 
        HumanMessage(content=combined_input),
    ]

    result = StrOutputParser().invoke(model.invoke(messages))
    
    return jsonify({'response': result})

if __name__ == '__main__':
    # Test using a default question
    #test_query = "Hi"
    test_query = "My landlord is not returning the security deposit after I vacated the property. What should I do?"
    #test_query = "What are my legal rights in case I am living in an illegally built property. I have received a notice from municipality in order to vacate the property as soon as possible for demolition."
    #test_query = "Who is the prime minister of India?"
    with app.test_request_context('/get_advice', method='POST', json={'query': test_query}):
        response = get_advice()
        print("Test Response:", response.get_json().get('response', 'No response available'))

    # Run the Flask app
    app.run(debug=True, use_reloader=False)
