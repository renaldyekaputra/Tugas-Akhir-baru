import streamlit as st
import plotly_express as px
import pandas as pd
from pathlib import Path
import pickle
import logging



# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# Judul Halaman
st.set_page_config(
    page_title="Klasifikasi Data ISPU",
    page_icon="📈",
)

# Setup selecting csv file
file_select = st.sidebar.selectbox(
    label="Pilih file",
    options=[
                "Jakarta","Forecasting DKI1", "Forecasting DKI2", "Forecasting DKI3", "Forecasting DKI4", "Forecasting DKI5"
            ]
)

global df

if file_select == "Jakarta":
    uploaded_file = '../generated/website.csv'
if file_select == "Forecasting DKI1":
    uploaded_file = '../generated/DKI1_Forecasting.xlsx'
if file_select == "Forecasting DKI2":
    uploaded_file = '../generated/DKI2_Forecasting.xlsx'
if file_select == "Forecasting DKI3":
    uploaded_file = '../generated/DKI3_Forecasting.xlsx'
if file_select == "Forecasting DKI4":
    uploaded_file = '../generated/DKI4_Forecasting.xlsx'
elif file_select == "Forecasting DKI5":
    uploaded_file = '../generated/DKI5_Forecasting.xlsx'

if uploaded_file is not None:
    # print(uploaded_file)
    # print("hello")

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_csv(uploaded_file)

lin_model=pickle.load(open('lin_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
##svm=pickle.load(open('svm.pkl','rb'))
svm=pickle.load(open('../generated/model_svm_rbf.pkl','rb'))
svm_polinomial=pickle.load(open('../generated/model_svm_polinomial.pkl', 'rb'))

def classify(num):
    logging.debug('Prediction num value: ')
    logging.debug(num)
    return num[0]

def main():
    st.title("Website")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Air Pollution Classification</h2>
    </div>
    <div></div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['SVM - RBF','SVM - polinomial']
    option=st.sidebar.selectbox('Pilih model yang akan digunakan?',activities)
    
    st.markdown(
        """

        """
    )

    global numeric_columns
    global non_numeric_columns
    try:
        st.write(df)
        date_column = list(df.select_dtypes(['datetime']).columns)
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        # print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")
    
    hari = st.number_input('Masukkan hari', min_value=0, max_value=500, step=1)
    st.write('Hari yang dipilih ', hari)

    pm10 = df["PM10"].loc[hari]
    so2 = df["SO2"].loc[hari]
    co = df["CO"].loc[hari]
    o3 = df["O3"].loc[hari]
    no2 = df["NO2"].loc[hari]

    st.subheader(option)
    st.write('PM10 : ', pm10)
    st.write('SO2 : ', so2)
    st.write('CO : ', co)
    st.write('O3 : ', o3)
    st.write('NO2 : ', no2)

    inputs=[[pm10, so2, co, o3, no2]]
    if st.button('Classify'):
        logging.debug('Kernel ' + option)
        logging.debug('Inputs ', inputs)
 #       if option=='Linear Regression':
 #           st.success(classify(lin_model.predict(inputs)))
 #       elif option=='Logistic Regression':
 #           st.success(classify(log_model.predict(inputs)))
        if option=='SVM - RBF':
           st.success(classify(svm.predict(inputs)))
        elif option=='SVM - polinomial':
            st.success(classify(svm_polinomial.predict(inputs)))
        
if __name__=='__main__':
    main()
