import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load data yang sudah diproses
df = pd.read_csv('processed_data.csv')

# Inisialisasi spaCy untuk normalisasi
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    # Ubah ke lowercase
    text = text.lower()
    
    # Hapus karakter khusus dan simbol
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalisasi kontraksi (contoh sederhana)
    text = re.sub(r'\(quote ll\)', 'will', text)
    text = re.sub(r'\(quote re\)', 'are', text)
    
    # Hapus whitespace berlebih
    text = ' '.join(text.split())
    
    return text

def normalize_grammar(text):
    doc = nlp(text)
    # Contoh normalisasi: ubah angka menjadi digit
    return ' '.join([
        token.text if not token.like_num 
        else token.text.replace(token.text, str(token.shape_))
        for token in doc
    ])

# Proses cleaning
df['cleaned_text'] = df['TEXT'].apply(clean_text)
df['normalized_text'] = df['cleaned_text'].apply(normalize_grammar)

# Ekstraksi fitur dengan TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['normalized_text'])
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# Simpan hasil
df.to_csv('data/cleaned_data.csv', index=False)
df_tfidf.to_csv('data/tfidf_features.csv', index=False)

print("Preprocessing selesai! File disimpan di folder data/processed/")