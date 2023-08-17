import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex

# Load the data from the CSV file
csv_filename = "applavialinks.csv"
df = pd.read_csv(csv_filename)

# Preprocess the text data (URL, Title, and Description)
def preprocess_text(text):
    return text.lower()

df["URL"] = df["URL"].apply(preprocess_text)
df["Title"] = df["Title"].apply(preprocess_text)
df["Description"] = df["Description"].apply(preprocess_text)

# Combine Title and Description for better representation
df["Title_Desc"] = df["Title"] + " " + df["Description"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Title_Desc"])

# Create an Annoy index for fast approximate nearest neighbors search
annoy_index = AnnoyIndex(tfidf_matrix.shape[1], "angular")
for i in range(tfidf_matrix.shape[0]):
    annoy_index.add_item(i, tfidf_matrix[i].toarray()[0])
annoy_index.build(50)  # 50 trees for the index

# Streamlit UI
st.title("Link Similarity Search")

# User input: link or title
user_input = st.text_input("Input:", "Enter a link or a title")

# Search button
if st.button("Search"):
    user_vector = vectorizer.transform([user_input]).toarray()[0]
    similar_indices = annoy_index.get_nns_by_vector(user_vector, 5)
    st.write("Suggested similar links:")
    for idx in similar_indices:
        st.markdown(f"[{df.iloc[idx]['Title']}]({df.iloc[idx]['URL']}) - {df.iloc[idx]['Description']}")