import streamlit as st
import pandas as pd
import pickle 
import numpy
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import preprocessing as pre
import st_model as stm 

@st.cache(allow_output_mutation=True)
def load_model(path:str):
    """
    Load model from model path.

    Args: 
        path (str): Path to model.

    Returns:
        model (object): Previously pickled model.
    """
    # Open pickle and load model
    pickle_in = open(path, 'rb')
    model = pickle.load(pickle_in)

    return  model

def load_image(path_to_image:str = r'D:\Programing projects\Master_rad\images\final_pic.jpg'):
    """
    Load image from path.

    Args:
        path_to_image (str): Path to image.

    Returns:
        image (object): Image from path.
    """
    image = Image.open(path_to_image)
    st.image(image, width=700)

    return image

def load_data_frame(path:str = "D:\Programing projects\Master_rad\data\Podaci_za_sve_masine.xlsx"):
    """
    Load data from path.

    Args:
        path (str): Path to data.

    Returns:
        df (pd.DataFrame): Data from path.
    """
    data_frame = pd.read_excel(path)

    return data_frame

# predvidjanje
def predict_fun(model:object,
                input_to_model:pd.DataFrame):
    """
    Takes input from user and returns prediction and probability.

    Args:  
        input_to_model (pd.DataFrame): Input from user.
        model (object): Model to use for prediction.

    Returns:
        prediction (str): Prediction from model.
        probability (float): Probability of prediction.
    """
    prediction = model.predict(input_to_model)
    prediction_proba = model.predict_proba(input_to_model)

    return prediction, prediction_proba

def preprocessing_of_input_data(df: pd.DataFrame):
    """
    Preprocess input data. 
    Put categorical input into one hot encoder and scale numerical data.
    
    Args:
        df: pd.DataFrame - User input data.

    Returns:
        combined_data: pd.DataFrame - Preprocessed input data.
    """
    #Get numerical and categorical features
    numerical = df['Vreme'] 
    
    categorical = ['Masina', 'RAD', 'Vrsta Masine', 'Uzrok', 'Tehnoloski','Elektro/struja', 'Mehanicki',
                   'Zloupotreba', 'Organizacioni', 'Eksterni uticaj']
    
    # Load and apply one-hot encoder to the relevant columns
    OH_encoder = load_model('D:\Programing projects\Master_rad\pickled_models\one_hot_encoder_8_14.pkl')
    OH_cols = pd.DataFrame(OH_encoder.transform(df[categorical]))
    
    # Replace default column names with more descriptive ones
    OH_cols.columns = OH_encoder.get_feature_names(categorical)

    # One-hot encoding removed index; put it back
    OH_cols.index = df.index
    
    # StandardScaler
    numerical = numerical.values
    scaler = load_model('D:\Programing projects\Master_rad\pickled_models\standard_scaler_8_14.pkl')
    numerical_scaled = pd.DataFrame(scaler.transform(numerical.reshape(-1, 1)))

    combined_data = pre.oh_and_scaled_data(OH_cols, numerical_scaled)
    
    return combined_data

# Ovde pocinje main:
def user_inut_features():
    st.sidebar.title('Predvidjanje Stepena Opasnosti Kod Zastoja')
    
    # unosenje informacija
    
    Masina = str.lower(st.sidebar.text_input('Masina', 'Unesi naziv masine'))
    RAD =  str.lower(st.sidebar.text_input('Rad', 'Unesi RAD'))
    Vrsta_Masine =  str.lower(st.sidebar.text_input('Vrsta Masine', 'Unesi Vrstu Masine'))
    Vreme = round(st.sidebar.number_input('Unesi vreme zastoja'))
    Uzrok =  str.lower(st.sidebar.text_input('Uzrok', 'Unesi uzrok zastoja'))
    Vrste_zastja = st.sidebar.multiselect(
        'Odaberite vrste zastja',
        ['Tehnoloski', 'Elektro/struja', 'Mehanicki', 'Zloupotreba', 'Organizacioni', 'Eksterni uticaj']
    )
    if 'Tehnoloski' in Vrste_zastja:
        Tehnoloski = 1
    else:
        Tehnoloski = 0
        
    if 'Elektro/struja' in Vrste_zastja:
        Elektro_struja = 1
    else:
        Elektro_struja = 0
        
    if 'Mehanicki' in Vrste_zastja:
        Mehanicki = 1
    else:
        Mehanicki = 0

    if 'Zloupotreba' in Vrste_zastja:
        Zloupotreba = 1
    else:
        Zloupotreba = 0
        
    if 'Organizacioni' in Vrste_zastja:
        Organizacioni = 1
    else:
        Organizacioni = 0  
    
    if 'Eksterni uticaj' in Vrste_zastja:
        Eksterni_uticaj = 1
    else:
        Eksterni_uticaj = 0
        
        
    features = pd.DataFrame({'Masina': Masina, 'RAD': RAD, 'Vrsta Masine': Vrsta_Masine, 'Vreme': Vreme,
                      'Uzrok': Uzrok, 'Tehnoloski': Tehnoloski, 'Elektro/struja': Elektro_struja,
                      'Mehanicki': Mehanicki, 'Zloupotreba': Zloupotreba, 'Organizacioni': Organizacioni,
                      'Eksterni uticaj': Eksterni_uticaj}, index=[0])
    
    return features

pd.set_option('display.max_colwidth', -1)


if __name__ == '__main__':
    
    STYLE = """
        <style>
        img {
             width="100" 
             height="125";
        }
        </style>
        """    
    #st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    
    load_image()
    
    st.write("""
    # PREDVIDJANJE STEPENA OPASNOSTI KOD ZASTOJA
    ***
    """)

    # load data on wich we train our models
    data_frame = load_data_frame()
    st_dataframe = stm.prepare_data_for_similarity_comparison(data_frame) # prepare data for sentence transformaer model

    # load random forest model
    random_forest_model = load_model('D:\Programing projects\Master_rad\pickled_models\Random_Forest_8_17.pkl')

    # Get user input into a data frame
    user_input = user_inut_features()

    # inport sentence transformer model
    sentence_transformers_model = load_model('D:\Programing projects\Master_rad\pickled_models\SentenceTransformer_8_19.pkl')

    st_encoding_list = load_model('D:\Programing projects\Master_rad\pickled_models\st_encoding_list_8_21.pkl') # get sentence encodings for all data

    # Get sentence transformer prediction to use as input for random forest model
    sentence_prediction_of_the_best_match_for_RAD = stm.get_sentence_transformers_prediction(st_dataframe, sentence_transformers_model, user_input, 
                                                                                                st_encoding_list, encoding_list_index=0, column_name='RAD')
    sentence_prediction_of_the_best_match_for_Uzrok = stm.get_sentence_transformers_prediction(st_dataframe, sentence_transformers_model, user_input, 
                                                                                                st_encoding_list, encoding_list_index=1, column_name='Uzrok')

    # Pass the sentence transformer prediction to the data frame for random forest model

    user_input['RAD'] = sentence_prediction_of_the_best_match_for_RAD
    user_input['Uzrok'] = sentence_prediction_of_the_best_match_for_Uzrok

    data_for_the_random_forest_model = preprocessing_of_input_data(user_input)
        
    if st.button('Predvidi'):
        result, result_proba = predict_fun(random_forest_model, data_for_the_random_forest_model)
        st.subheader('Karakteristike koje je uneo koriskik')
        st.write(user_input)
        # streamlit display results 
        st.markdown('## Stepen opasnosti je:', unsafe_allow_html=True)
        st.markdown(f'## <center>Opasnost: **{result[0]}**</center>', unsafe_allow_html=True)        
        # streamlit display results_probability
        st.markdown('## Verovatnoca predvidjanja je:', unsafe_allow_html=True)
        verovatnoca = pd.DataFrame(result_proba, columns=('S.O. %d' % i for i in [1,2,3,4,5])) 
        st.dataframe(verovatnoca)