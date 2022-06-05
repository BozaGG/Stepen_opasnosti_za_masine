import streamlit as st
import pandas as pd
import pickle 
import numpy
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

def load_model(path: str = 'RandomForest.pkl'):
    """Load model model from model path"""
    # Open pickle and load model
    pickle_in = open(path, 'rb')
    return pickle.load(pickle_in)

def load_image(path_to_image: str = 'download'):
        image = Image.open(path_to_image)
        st.image(image, width=700)

# predvidjanje
def predict_fun(input_to_model:pd.DataFrame):
    """
    input_to_model - 
    """
    prediction = model.predict(input_to_model)
    prediction_proba = model.predict_proba(input_to_model)
    return prediction, prediction_proba

# Staviti kategoricke podatke u One-Hots encoder
def preprocessing_of_data(df: pd.DataFrame):
    
    #Get numerical and categorical features
    numerical = df['Vreme'] # vreme koliko je trajao zastoj
    
    categorical = ['Masina', 'RAD', 'Vrsta Masine', 'Uzrok', 'Tehnoloski','Elektro/struja', 'Mehanicki',
                   'Zloupotreba', 'Organizacioni', 'Eksterni uticaj']
    
    # Apply one-hot encoder to the relevant columns
    OH_encoder = load_model('encder.pkl')
    OH_cols = pd.DataFrame(OH_encoder.transform(df[categorical]))
    
    # Replace default column names with more descriptive ones
    OH_cols.columns = OH_encoder.get_feature_names(categorical)

    # One-hot encoding removed index; put it back
    OH_cols.index = df.index
    
    # StandardScaler
    numerical = numerical.values
    scaler = load_model('StandardScaler_15_5_2022.pkl')
    numerical_scaled = pd.DataFrame(scaler.transform(numerical.reshape(-1, 1)))
    
    OH_cols['Vreme'] = numerical_scaled
    
    X = OH_cols
    
    return X

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
    
    load_image('final_pic.jpg')
    
    st.write("""
    # PREDVIDJANJE STEPENA OPASNOSTI KOD ZASTOJA
    ***
    """)
    
    #file = st.file_uploader("Upload-ujte fajl za vise istovremenih predvidjanja", type=["csv", "xlsx"])
    #show_file = st.empty()

    model = load_model(path='RandomForest.pkl')
    df = user_inut_features()

    #st.subheader('Karakteristike koje je uneo koriskik')
    #st.write(df)

    podaci_za_alg = preprocessing_of_data(df)
        
    if st.button('Predvidi'):
        result, result_proba = predict_fun(podaci_za_alg)
        st.subheader('Karakteristike koje je uneo koriskik')
        st.write(df)
        # streamlit display results 
        st.markdown('## Stepen opasnosti je:', unsafe_allow_html=True)
        st.markdown(f'## <center>Opasnost: **{result[0]}**</center>', unsafe_allow_html=True)        
        # streamlit display results_probability
        st.markdown('## Verovatnoca predvidjanja je:', unsafe_allow_html=True)
        verovatnoca = pd.DataFrame(result_proba, columns=('S.O. %d' % i for i in [1,2,3,4,5])) 
        st.dataframe(verovatnoca)