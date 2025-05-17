import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Dataset baru: Kumpulan teks dengan label sentimen (positive/negative)
texts = [
    "The movie was fantastic! I loved it.",
    "Absolutely terrible experience, never again.",
    "The service was excellent and very professional.",
    "I hated the food, it was disgusting.",
    "What a wonderful day, everything went perfectly.",
    "The product broke after one use, very disappointing.",
    "I am so happy with my purchase, highly recommend!",
    "Worst customer service I have ever encountered."
]

# Label sentimen untuk setiap teks
labels = [
    "positive",  # Sentimen positif
    "negative",  # Sentimen negatif
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative"
]

# Fungsi preprocessing untuk membersihkan teks
def preprocess_text(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-z\s]', '', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Terapkan preprocessing ke seluruh dataset
texts = [preprocess_text(text) for text in texts]

# Membagi dataset menjadi data latih (70%) dan data uji (30%)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Membuat pipeline: TF-IDF untuk representasi teks + SVM untuk klasifikasi
model = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Mengubah teks menjadi vektor numerik menggunakan TF-IDF
    ('svm', SVC(kernel='linear'))  # Model SVM dengan kernel linear
])

# Melatih model menggunakan data latih
model.fit(X_train, y_train)

# Mengevaluasi model menggunakan data uji
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))  # Menampilkan metrik evaluasi seperti precision, recall, dan F1-score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  # Menampilkan akurasi model

# Menguji model dengan kalimat baru
test_sentences = [
    "I am extremely happy with the results!",  # Kalimat positif
    "This is the worst experience of my life.",  # Kalimat negatif
    "Fantastic job, well done!",  # Kalimat positif
    "I regret buying this product, horrible quality.",  # Kalimat negatif
    "Absolutely loved it, great experience."  # Kalimat positif
]

# Preprocessing dan prediksi untuk kalimat baru
print("\nPredictions on new sentences:")
for sentence in test_sentences:
    processed_sentence = preprocess_text(sentence)  # Preprocessing kalimat
    prediction = model.predict([processed_sentence])[0]  # Prediksi sentimen
    print(f"'{sentence}' => {prediction}")