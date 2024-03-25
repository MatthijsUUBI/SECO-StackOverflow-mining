import regex as rx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load NLTK stopwords
stop_words = set(stopwords.words("english"))

# Initialize Porter Stemmer
ps = PorterStemmer()

# Step 1: Filter the strings
def filter_content(content: str):
    # remove code bits
    result = rx.sub(r"<pre.*?</pre>", "", content, flags=rx.DOTALL)
    # remove HTML tags
    result = rx.sub(r"<.*?>", "", result)
    # remove special characters
    result = rx.sub(r"[^\w\s]", "", result)
    # decapitalize
    result = result.lower()
    # remove extra whitespaces
    result = result.strip()
    return result

# Step 2: Remove stopwords and stem words
def preprocess_text(content: str):
    word_tokens = word_tokenize(content)
    result = []
    for w in word_tokens:
        if w not in stop_words:
            stemmed_word = ps.stem(w)
            result.append(stemmed_word)
    return result

# Preprocessing function
def preprocess(df):
    # Filter the text of each cell
    df['Body'] = df['Body'].apply(filter_content)
    # Remove stopwords and stem words
    df['Preprocessed Content'] = df['Body'].apply(preprocess_text)
    return df['Preprocessed Content']
