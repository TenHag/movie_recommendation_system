import streamlit as st
import pandas as pd
import requests
import pickle

def fetch(movie_id):
    response_API = requests.get("https://api.themoviedb.org/3/movie/" + str(movie_id) + "?api_key=83abc4cd1a30787f6b445bee1cfed5a4")

    data = response_API.json()
    return "https://image.tmdb.org/t/p/w500/"  + data['poster_path']


def recommend(movie):
    inox = movies[movies['title'] == movie].index[0]
    dist = sim[inox]
    movie_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies=[]
    recommended_movies_posters=[]

    for i in movie_list:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch(movie_id))

    return recommended_movies,recommended_movies_posters


movies_dict =pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
st.title('Movie Recommendation System')

sim=pickle.load(open('similarity.pkl','rb'))

option =st.selectbox(' Enter the movie name: ',movies['title'].values
                      )
if st.button('Recommend'):
    names,posters=recommend(option)
    col1,col2,col3,col4,col5= st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
