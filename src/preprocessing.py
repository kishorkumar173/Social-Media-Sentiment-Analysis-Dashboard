import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download required resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # lowercase
    text = text.lower()
    
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # remove mentions (@username)
    text = re.sub(r"@\w+", "", text)
    
    # remove hashtags (#tag → tag)
    text = re.sub(r"#", "", text)
    
    # remove special characters & numbers
    text = re.sub(r"[^a-z\s]", "", text)
    
    # remove stopwords + lemmatization
    words = []
    for word in text.split():
        if word not in stop_words:
            word = lemmatizer.lemmatize(word)
            words.append(word)
    
    return " ".join(words)