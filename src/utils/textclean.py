# textclean.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# One-time downloads (put this in your notebook or first run script)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def clean_for_embedding(text: str) -> str:
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words and t not in punctuation and t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)