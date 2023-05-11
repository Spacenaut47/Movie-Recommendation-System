import gradio as gr
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to recommend similar movies
def recommend_movies(movie_name):
    # Loading the data from the CSV file to a pandas dataframe
    movies_data = pd.read_csv('movies.csv')

    # Selecting the relevant features for recommendation
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Replacing the missing values or the null values with null string using for loop
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combining all the selected features
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

    # Converting the text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Similarity using cosine similarity algorithm
    similarity = cosine_similarity(feature_vectors)

    # Finding the close match for the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist())
    if len(find_close_match) > 0:
        close_match = find_close_match[0]
        # Finding the index of the movie with title
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        # Getting a list of similar movies
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        # Sorting the movies based on their similarity score
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        # Return the recommended movies as a list
        recommended_movies = [movies_data[movies_data.index == movie[0]]['title'].values[0] for movie in sorted_similar_movies]
        return "\n".join(recommended_movies)
    else:
        return "No close match found for the given movie."

# Create the Gradio interface
iface = gr.Interface(fn=recommend_movies, inputs="text", outputs="text", title="Movie Recommendation System")
iface.launch()
