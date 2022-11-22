from PIL import Image
import requests
import streamlit as st
from streamlit_lottie import st_lottie

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="COVID-19 Analysis",page_icon="virus.png",layout="wide")

def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()


# load assets
lottie_coding=load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vckswclv.json")
img_contact_form=Image.open("images/c.jpeg")
img_lottie_animation=Image.open("images/doctor.png")
 
 
 #---header---
with st.container():
    st.title("COVID-19 DATA ANALYSIS")
    st.subheader("Introduction")
    st.write('''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus./n
Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. 
The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.
The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.
''')
    st.write("To explore more click on below!")
    st.write("[Learn_More >](https://www.aarogyasetu.gov.in/)")

#what to do
with st.container():
    st.write("---")
    left_column, right_column= st.columns(2)
    with left_column:
        st.header("Division of Datasets:")
        st.subheader("1.Country wise Covid cases till June 2022")
        st.write('''-having attributes country ,total_confirmed, total_deaths, total_recovery ,active cases, serious_or_critical
''')
        
        st.subheader("2.State wise Covid cases of India till June 2022")
        st.write("-having attributes such as Name of state/UTs, Total Cases, Active Cases,Discharged and Deaths.")
        st.subheader('''3.Dataset for the prediction of Covid-19 based on symptoms ''')
        st.write("-having initial clinical symptoms of covid-19")
with right_column:
    st_lottie(lottie_coding, height="200",key="covid-19")

#--project
with st.container():
    st.write("---")
    st.header("Visualization of Cases")
    st.write()
    image_column, text_column=st.columns((1,2))
    with image_column:
        st.image(img_contact_form)
    with text_column:
        st.subheader("State Wise Visualization of Indian States")
        st.write("See the records of 28 Indian states and 8 union Territories...")
    image = Image.open('Indian.png')

st.image(image)

with st.container():
    image_column, text_column=st.columns((1,2))
    with image_column:
        st.image(img_lottie_animation)
    with text_column:
        st.subheader("Country Wise Visualization of World")
        st.write("See the records of 227 Countries of the World...")
image = Image.open('A.png')

st.image(image)


with st.container():
    st.header("Prediction")
    covid_data = pd.read_csv('testing.csv')
    x=covid_data[covid_data.columns[0:5]]
    x=np.array(x)
    y=covid_data["corona_result"]
    y=np.array(y)
    y=y.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    name=st.text_input("Enter Your Full Name: ")
    dept=st.text_input("Enter the Name of your department/Purpose to come: ")
    add=st.text_input("Enter your address: ")
    time=st.text_input("Enter time in HH:MM: ")
    con=st.text_input("Any Recent travel history: ")

    st.header("NOTE:Please enter 0 if you are not suffering with any of the mentioned symptoms and 1 if you are suffering.")
    user_values=[]
    input_value1=st.text_input("Are you suffering from cough? :")
    user_values.append((input_value1))
    input_value2=st.text_input("Are you suffering from fever? :")
    user_values.append((input_value2))
    input_value3=st.text_input("Do you have sore throat? :")
    user_values.append((input_value3))
    input_value4=st.text_input("Are you suffering from Shortness of breath? :")
    user_values.append((input_value4))
    input_value5=st.text_input("Do you have headache :")
    user_values.append((input_value5))
    user_values=[user_values]

    agree=st.button("Predict")
    if agree:
        result=model.predict(user_values)
    
        if result=='0':
            st.write(name,"you are not affected with Covid-19!  You can enter...")
        else:
            st.write(name,"you are showing the symptoms of Covid-19! Get your RTPCR test done as soon as possible... ")
#--contact--
with st.container():
    st.subheader("Thank you for giving your valuable time!")
    