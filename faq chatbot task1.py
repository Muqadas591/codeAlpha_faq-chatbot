import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Initialize NLTK and spaCy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load and preprocess the dataset
file_path = 'C:\\Users\\test\\Downloads\\archive (1)\\Kaggle related questions on Qoura - Questions.csv'  # Adjust this path if necessary
df = pd.read_csv(file_path)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)

df['Processed_Question'] = df['Questions'].apply(preprocess_text)  # Adjust column name if necessary

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Question'])

def find_most_similar(query, X, vectorizer):
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarities = cosine_similarity(query_vec, X)
    return similarities.argmax()

def get_response(user_input):
    idx = find_most_similar(user_input, X, vectorizer)
    answered = df.iloc[idx]['Answered']  # Adjust column name if necessary
    link = df.iloc[idx]['Link']  # Adjust column name if necessary
    return f"Answer: {answered}\nMore info: {link}"

# Initialize the Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return """
    <!doctype html>
    <html>
    <head><title>Chatbot</title></head>
    <body>
        <h1>Chatbot</h1>
        <input type="text" id="user_input" placeholder="Type your message here..." />
        <button onclick="sendMessage()">Send</button>
        <p id="response"></p>
        <script>
            function sendMessage() {
                const userInput = document.getElementById('user_input').value;
                fetch(`/get?msg=${userInput}`)
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('response').innerText = data;
                    });
            }
        </script>
    </body>
    </html>
    """

@app.route("/get")
def get_bot_response():
    user_input = request.args.get('msg')
    return get_response(user_input)
if __name__ == "__main__":
    app.run(port=5000)




