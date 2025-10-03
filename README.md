### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 25.09.2025
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')

###### Sample documents stored in a dictionary
    documents = {
        "doc1": "This is the first document.",
        "doc2": "This document is the second document.",
        "doc3": "And this is the third one.",
        "doc4": "Is this the first document?",
    }

###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in               string.punctuation]
        return " ".join(tokens)

###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

###### Construct TF
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform(preprocessed_docs.values())
    tf_df = pd.DataFrame(tf_matrix.toarray(), index=preprocessed_docs.keys(), columns=count_vectorizer.get_feature_names_out())
    print("=== Term Frequency (TF) ===")
    print(tf_df)

###### Display TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=preprocessed_docs.keys(), columns=tfidf_vectorizer.get_feature_names_out())
    print("\n=== TF-IDF ===")
    print(tfidf_df.round(4))
    
###### Calculate cosine similarity between query and documents
    def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vec = tfidf_vectorizer.transform([preprocessed_query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = []
    doc_ids = list(preprocessed_docs.keys())
    for i, score in enumerate(cosine_sim):
        results.append((doc_ids[i], documents[doc_ids[i]], score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results
    
###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = search(query, tfidf_matrix, tfidf_vectorizer)

###### Display search results
    print("Query:", query)
    for i, result in enumerate(search_results, start=1):
        print(f"\nRank: {i}")
        print("Document ID:", result[0])
        print("Document:", result[1])
        print("Similarity Score:", result[2])
        print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("The highest rank cosine score is:", highest_rank_score)

### Output:
<img width="570" height="736" alt="image" src="https://github.com/user-attachments/assets/c2bc8b65-874f-48e7-a74f-ff30b074c772" />

### Result:
Thus the python program to find Information Retrieval Using Vector Space Model is executed successfully.
