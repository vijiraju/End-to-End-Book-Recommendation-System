import os
import sys
import pickle
import streamlit as st
import numpy as np
from book_recommender.logger.log import logging
from book_recommender.config.configuration import AppConfiguration
from book_recommender.pipeline.training_pipeline import TrainingPipeline
from book_recommender.exception.exception_handler import AppException

class Recommendation:
    def __init__(self,app_config=AppConfiguration()):
        try:
            self.recommendation_config=app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e,sys) from e


    def fetch_poster(self,suggestion):
        try:
            book_name=[]
            ids_index=[]
            poster_url=[]
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            final_rating =  pickle.load(open(self.recommendation_config.final_rating_serialized_objects,'rb'))
            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])
            for name in book_name[0]:
                ids=np.where(final_rating['title']==name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url=final_rating.iloc[idx]['image_url']
                poster_url.append(url)
            return poster_url


        except Exception as e:
            raise AppException(e,sys) 
    def recommend_book(self,book_name):
        try:
            books_list = []
            model = pickle.load(open(self.recommendation_config.trained_model_path,'rb'))
            book_pivot =  pickle.load(open(self.recommendation_config.book_pivot_serialized_objects,'rb'))
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )

            poster_url = self.fetch_poster(suggestion)
            
            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list , poster_url   
        
        except Exception as e:
            raise AppException(e, sys) from e
        
    def train_engine(self):
        try:
            obj=TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training completed")
            logging.info("reccomend successfull")

        except Exception as e:
            raise AppException(e,sys) from e
    def recommendations_engine(self,selected_books):
        try:
            self.recommend_book,poster_url=self.recommend_book(selected_books)
            col1,col2,col3,col4,col5=st.columns(5)
            with col1:
                st.text(self.recommend_book[1])
                st.image(poster_url[1])
            with col2:
                st.text(self.recommend_book[2])
                st.image(poster_url[2])

            with col3:
                st.text(self.recommend_book[3])
                st.image(poster_url[3])
            with col4:
                st.text(self.recommend_book[4])
                st.image(poster_url[4])
            with col5:
                st.text(self.recommend_book[5])
                st.image(poster_url[5])
        except Exception as e:
            raise AppException(e, sys) from e
        
if __name__=="__main__":
    st.header("ENd to end recommendation")
    st.text("colloborative filtering")
    obj=Recommendation()

    if st.button('Train recommder system'):
        obj.train_engine()

    book_names=pickle.load(open(os.path.join('templates','book_names.pkl') ,'rb'))
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names)
    
    #recommendation
    if st.button('Show Recommendation'):
        obj.recommendations_engine(selected_books)



        
    
        







