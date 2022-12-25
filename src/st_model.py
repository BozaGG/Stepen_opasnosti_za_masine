import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import preprocessing as pre
import pickle
import datetime
import streamlit as st

@st.cache()
def get_st_model(model_path:str = 'paraphrase-multilingual-mpnet-base-v2') -> SentenceTransformer:
    """
    Returns a SentenceTransformer model for NER.

    Args: 
        model_path:str - Path to the model.

    Returns:
        SentenceTransformer - The model.
    """
    return SentenceTransformer(model_path)

@st.cache()
def get_st_encodings(model:SentenceTransformer, 
                        data_frame:pd.DataFrame,
                        pickle_sentence_transformer:bool = False,
                        pickle_file_name:str = 'SentenceTransformer') -> pd.DataFrame:
    """
    Returns the encodings of the sentences in a list.
    If pickle_sentence_transformer is True, the encodings are pickled and saved in a file with the name pickle_file_name.
    Also pickles the st_encoding_list.
    
    Args:
        model:SentenceTransformer - The model.
        data_frame:pd.DataFrame - The data frame with sentences to be encoded.
        pickle_sentence_transformer:bool - Whether to pickle the sentence transformer.
        pickle_file_name:str - The name of the pickle file to save the sentence transformer.
    
    Returns:
        st_encoding_list - The encoding of the sentences in a list. Where the first element is the encoding of the first sentence, 
                            the second element is the encoding of the second sentence and so on.
    """
    st_encoding_list = list()
    for column in data_frame.columns:
            st_encoding_list.append(model.encode(data_frame[column].unique()))

    if pickle_sentence_transformer == True:
        now = datetime.datetime.now()
        pickle_transformer = open('{}_{}_{}.pkl'.format(pickle_file_name, now.month, now.day), 'wb')
        pickle.dump(model, pickle_transformer)
        pickle_transformer.close()

        pickle_encoding_list = open('{}_{}_{}.pkl'.format('st_encoding_list', now.month, now.day), 'wb')
        pickle.dump(st_encoding_list, pickle_encoding_list)
        pickle_encoding_list.close()

    return st_encoding_list

def get_st_similarity(st_encoding_list:list,
                        user_input_sentence:str,
                        model:SentenceTransformer) -> float:
    """
    Returns the similarity between the user input sentence and the other sentences in the list.
    
    Args:
        st_encoding_list:list - The list of encodings of the sentences. Must chose the column with the right encoding. [0] for 'RAD', [1] for 'Uzrok'.
        user_input_sentence:str - The sentence to be compared with the other sentences.
        model:SentenceTransformer - The sentence transformer model.
    
    Returns:
        similarity:float - The list of similarities between the user input sentence and the other sentences in the list.
    """
    evaluating_sentence = model.encode(user_input_sentence) # Encode the user input sentence.

    similarity = cos_sim(evaluating_sentence, st_encoding_list) 

    return similarity

def get_the_best_match(data_frame_column:pd.DataFrame,
                        similarity:list):
    """
    Rturns the sentence with the highest similarity to the user input sentence.

    Args:
        similarity:list - The list of similarities between the user input sentence and the other sentences in the list.
        data_frame_column_w_unique_values:pd.Series - The column with the unique values of the data frame.

    Returns:
        best_match:str - The sentence with the highest similarity.
    """

    data_frame_column_w_unique_values = data_frame_column.unique()  # Get the unique values of the data frame column.

    best_match = data_frame_column_w_unique_values[similarity.argmax()] # Get the sentence with the highest similarity.

    return best_match

def prepare_data_for_similarity_comparison(data_frame:pd.DataFrame,
                                            if_user_input_sentence:bool = False):
    """
    Prepare data for using functions from preprocessing.py
    if_user_input_sentence:bool - If True, the user input sentence is prepared for similarity comparison.

    Args: 
        data_frame:pd.DataFrame 

    Returns:
        prepared_data_frame:pd.DataFrame - The prepared data frame.
    """
    if if_user_input_sentence == False:
        data_frame = pre.remove_unwanted_coulmns(data_frame, columns= ['Rbr', 'Dan', 'Datum', 'Masina', 'MOTORNI SAT', 'VRSTA MASINE',
                                                            'Vreme', 'Tehnoloski', 'Elektro/struja', 'Mehanicki', 'Zloupotreba', 
                                                            'Organizacioni', 'Eksterni uticaj', 'Nor stepen opasnosti'])
    else:
        pass
    data_frame = pre.cyrilic_to_latin(data_frame, columns=['Uzrok', 'RAD'])
    data_frame = pre.lower_and_strip_all_data(data_frame, columns=['Uzrok', 'RAD'])
    prepared_data_frame = pre.fill_missing_values_based_on_similar_column(data_frame,
                                                            column_w_nan = ['Uzrok'],
                                                            columns_to_fill= ['RAD'])

    return prepared_data_frame
    
def transform_user_input_into_string(user_input_sentence:pd.DataFrame,
                                        column:str) -> str:
    """
    Transform pandas column into srting to later put into similarity function.

    Args:
        user_input_sentence:pd.DataFrame - The user input sentence.
        column:str - The column to be transformed.
    
    Returns:   
        string_input:str - The user input sentence in string format.
    """
    string_input = user_input_sentence[column].to_string(index=False)

    return string_input

def  get_sentence_transformers_prediction(data_frame:pd.DataFrame,
                                            model:SentenceTransformer,
                                            user_input_sentence:pd.DataFrame,
                                            st_encoding_list:list,
                                            encoding_list_index:int,
                                            column_name:str):
    """
    Returns the prediction of the model. The prediction is the sentence with the highest similarity to the user input sentence.
    
    Args:
        data_frame:pd.DataFrame - The data frame with the sentences to be predicted.
        model:SentenceTransformer - The sentence transformers model to be used.
        user_input_sentence:pd.DataFrame - The user input sentence.
        st_encoding_list:list - The list of encodings of the sentences. It is a list of lists where the first element is the encoding for RAD
                                                                                                 and the second element is the encoding for Uzrok.
        encoding_list_index:int - The index of the encoding list. (0 for 'RAD', 1 for 'Uzrok')
        column_name:str - The name of the column to be predicted.
        
    Returns:
        best_match:str - The sentence with the highest similarity.
     """
    processed_input = prepare_data_for_similarity_comparison(user_input_sentence, if_user_input_sentence=True)

    string_input = transform_user_input_into_string(processed_input, column_name)
    
    similarity = get_st_similarity(st_encoding_list[encoding_list_index], string_input, model)
    best_match = get_the_best_match(data_frame[column_name], similarity)

    return best_match

if __name__ == '__main__':
    data = pd.read_excel("D:\Programing projects\Master_rad\data\Podaci_za_sve_masine.xlsx")
    data = prepare_data_for_similarity_comparison(data)
    model = get_st_model()
    st_encoding_list = get_st_encodings(model, data)
    user_input = pd.DataFrame({'RAD': ['dugmici najbitnije elektro kutije krana'], 'Uzrok': ['dugmici najbitnije elektro kutije krana'], 'Rbr': [1], 'Dan': [1], 'Datum': [1], 'Masina': [1], 'MOTORNI SAT': [1], 'VRSTA MASINE': [1], 'Vreme': [1], 'Tehnoloski': [1], 'Elektro/struja': [1], 'Mehanicki': [1], 'Zloupotreba': [1], 'Organizacioni': [1], 'Eksterni uticaj': [1], 'Nor stepen opasnosti': [1]})
    data = get_sentence_transformers_prediction(data, model, user_input, st_encoding_list, encoding_list_index=0, column_name='RAD')
