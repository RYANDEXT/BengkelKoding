import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from streamlit_lottie import st_lottie
from sklearn.preprocessing import MinMaxScaler


#judul
st.write("""
# Klasifikasi Heart Disease (Gaussian Naive Bayes)
Aplikasi Berbasis Web Untuk Memprediksi (mengklasifikasi) Penyakit Jantung
Data yang didapat dari Repositori UCI Machine Learning
Dataset tersebut adalah dataset penyakit jantung yang terdiri dari 303 data individu
""")

#image

#lottie animation
def lottie_file():
    with open("Animation-1704105124757.json", 'r', encoding='utf-8') as p:
        return json.load(p)
    
path = lottie_file()
st_lottie(path)

#sidebar
st.sidebar.header('Parameter Inputan')

#input upload file csv untuk parameter inputan, ada inputan filem dan juga inputan manual
upload_file = st.sidebar.file_uploader("Upload Csv", type= "csv")

if upload_file is not None:
    inputan = pd.read_csv(upload_file)
else:
    def input_user():
        age = st.sidebar.slider('Age', 0,150,75)
        sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
        cp = st.sidebar.selectbox('CP', ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
        trestbps = st.number_input('Trestbps', 0, 250, 150)
        chol = st.sidebar.slider('Chol', 0,400,100)
        fbs = st.sidebar.selectbox('FBS', ('No', 'Yes'))
        restecg = st.sidebar.selectbox('Restecg', ('Normal', 'Having ST-T wave abnormality', 'showing probable or definite left venctrivular hypertrophy by Estes "criteria"'))
        thalach = st.sidebar.slider('Thalach', 0, 400, 150)
        exang = st.sidebar.selectbox('Exang', ('No','Yes'))
        oldpeak = st.sidebar.slider('Oldpeak', 0.0, 10.0, 5.0)

        if sex == 'Female':
            sex = 0
        elif sex == 'Male':
            sex = 1

        if cp == 'typical angina':
            cp = 1
        elif cp == 'atypical angina':
            cp = 2
        elif cp == 'non-anginal pain':
            cp = 3
        elif cp == 'asymptomatic':
            cp = 4

        if fbs == 'No':
            fbs = 0
        elif fbs == 'Yes':
            fbs = 1
        
        if restecg == 'Normal':
            restecg = 0
        elif restecg == 'Having ST-T wave abnormality':
            restecg = 1
        else: restecg = 2

        if exang == 'No':
            exang = 0
        else: exang = 1

        data = {
            'age':age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
        }

        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()



# import dataset
HeartDisease_raw = pd.read_csv('dfClean_hungarian.csv')

#memisahkan dataset atribut dan label
HeartDisease = HeartDisease_raw.drop(columns=['target'])
#menggabungkan inputan user kedalam dataset yang sudah dipisah labelnya
df = pd.concat([inputan, HeartDisease], axis=0)

frow = df[:1]

# menampilkan parameter hasil inputan
st.subheader('Parameter Input')

if upload_file is not None:
    st.write(df)
else:
    st.write('Masukkan file csv')
    st.write(df)

# load model Gaussian Naive Bayes Classifier
load_model = pickle.load(open('modelKNN_HeartDisease.pkl', 'rb'))

# terapkan KNN

prediksi = load_model.predict(frow)
prediksi_proba = load_model.predict_proba(frow)

st.subheader('Keterangan Label Kelas')
jenis_HeartDisease = np.array(['No Disease', 'Disease 1', 'Disease 2', 'Disease 3', 'Disease 4'])
st.write(jenis_HeartDisease)

st.subheader('Hasil Prediksi Klasifikasi Heart Disease')
st.write(jenis_HeartDisease[int(prediksi[0])])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Heart Disease)')
st.write(prediksi_proba)
