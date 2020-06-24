import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List


def get_tf_idf_values(training_data: pd.DataFrame):
    list_of_texts = training_data['text'].to_list()
    vectorizer = CountVectorizer()
    count_vector = vectorizer.fit_transform(list_of_texts)  # count vector for all training documents
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(count_vector)  # learns the global idf vector


    def calculate_tfidf_score(lyrics: str):
        doc_count_vector = vectorizer.transform([lyrics])  # this is a count vector for this particular document
        tfidf_vector = tfidf_transformer.transform(doc_count_vector)  # global tf-idf scores for all ngrams in this doc
        feature_names = vectorizer.get_feature_names()
        df = pd.DataFrame(tfidf_vector.T.todense(), index=feature_names, columns=["tfidf"])
        reduced_df = df.sort_values(by=['tfidf'], ascending=False).head(50)
        tfidf_score = round(reduced_df['tfidf'].mean() * 10, 3)
        return tfidf_score

    training_data['tf_idf_score'] = training_data['text'].apply(calculate_tfidf_score)
    return training_data
