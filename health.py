import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
              #DB management
import sqlite3
import hashlib
import joblib
conn=sqlite3.connect('data.db')
c=conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username=? AND password=?',(username,password))
    data=c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data=c.fetchall()
    return data

def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
    if generate_hashes(password)==hashed_text:
        return hashed_text
    return false
from matplotlib import gridspec
#import time
from plotly.figure_factory import create_distplot
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
#from dictionaries import Dict
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing
#from sklearn.utils import shuffle
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.decomposition import PCA #PRINCIPLE components ANALYSIS

Heart=pd.read_csv('heart_failure_clinical_records_dataset.csv')
Heart1=pd.read_csv('heart_failure_clinical_records_dataset.csv')
Heart['sex']=Heart['sex'].replace([0,1],['Female','Male'])
Heart['smoking']=Heart['smoking'].replace([0,1],['Non Smoking','Smoking'])
def main():
    #"login page"
    menu=["Home","Login","SignUp"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="Home":
        st.title("Heart Failure Analysis")
        components.html("""
        <header>

         </header>
         <h1>

         </h1>
        <style>
        header {color:Black;
        font-size:50px;
        text-align:center;
        }
        h1{
        color:Black;
        font-size:30px;
        text-align:center;
        }

        body  {
        background-image: url('https://images.ctfassets.net/yixw23k2v6vo/6BezXYKnMqcG4LSEcWyXlt/b490656e99f34bc18999f3563470eae6/iStock-1156928054.jpg?w=802&fm=jpg&fit=thumb&q=65&fl=progressive');
        background-repeat: no-repeat;
        background-attachment:fixed;
        background-size: cover;
        }

        </style>
        """,width=700, height=450, scrolling=False)

    if choice=="Login":
        st.subheader("Login Section")
        username=st.sidebar.text_input("UserName ")
        password=st.sidebar.text_input("Password ",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pass=generate_hashes(password)
            result=login_user(username,verify_hashes(password,hashed_pass))
            if result:
                st.success("Logged In as {} ".format(username))
                hide_streamlit_style = """
                            <style>
                            footer {visibility: hidden;}
                            </style>
                            """
                st.markdown(hide_streamlit_style, unsafe_allow_html=True)
                select=st.sidebar.selectbox('Menu',['Plot','Prediction'])

                                 #CHECK ALL THE CREATED PROFILES
                #user_result=view_all_users()
                #DB=pd.DataFrame(user_result,columns=["Username","Password"])
                #st.write(DB)
                Heart.loc[Heart['age'] >=40, 'Age_Categ'] = '40-45'
                Heart.loc[Heart['age'] >45, 'Age_Categ'] = '46-50'
                Heart.loc[Heart['age'] >50, 'Age_Categ'] = '51-55'
                Heart.loc[Heart['age'] >55, 'Age_Categ'] = '56-60'
                Heart.loc[Heart['age'] >60, 'Age_Categ'] = '61-65'
                Heart.loc[Heart['age'] >65, 'Age_Categ'] = '66-70'
                Heart.loc[Heart['age'] >70, 'Age_Categ'] = '71-75'
                Heart.loc[Heart['age'] >75, 'Age_Categ'] = '76-80'
                Heart.loc[Heart['age'] >80, 'Age_Categ'] = '81-85'
                Heart.loc[Heart['age'] >85, 'Age_Categ'] = '86-90'
                Heart_diabetes=Heart.groupby(['Age_Categ','DEATH_EVENT','sex','smoking','high_blood_pressure'],as_index=False)['diabetes'].sum()
                #choice=st.multiselect("Add Your Info To Analyze",["diabetes","smoking","sex","DEATH_EVENT","high_blood_pressure"])
                #st.write(Heart_diabetes)
                #sns.catplot(x="Age_Categ", y="Heart_diabetes", hue="sex", kind="bar", data=Heart)
                #st.pyplot()

                if select=="Plot":
                    st.title("Heart Disease Dashbords ")
                    st.markdown('Factors Affecting Heart disease')
                    Factor=st.sidebar.selectbox("Choose Your Factor",['Sex','Age','Factors'])
                    if Factor=="Sex":
                        Sex_Data=Heart.groupby(['sex'],as_index=False)['DEATH_EVENT'].sum()
                        fig = px.bar(Sex_Data, x="sex", y="DEATH_EVENT",color="sex",barmode="group",title="Number of Death Cases per Age Category",width=800, height=400)
                        fig.update_layout(
                        autosize=False,
                        yaxis=dict(
                        titlefont=dict(size=20))
                )
                        st.plotly_chart(fig)
                        if st.checkbox("Show Data"):
                            st.write(Sex_Data)

                    if Factor=="Age":
                        Age_Data=Heart.groupby(['Age_Categ','sex'],as_index=False)['DEATH_EVENT'].sum()
                        fig = px.bar(Age_Data, x="Age_Categ", y="DEATH_EVENT",color="sex",barmode="group",title="Number of Death Cases per Age Category",width=800, height=400)
                        fig.update_layout(
                        autosize=False,
                        yaxis=dict(
                        titlefont=dict(size=20))
                )

                        st.plotly_chart(fig)
                        if st.checkbox("Show Data"):
                            st.write(Age_Data)
                    if Factor=="Factors":
                        if st.checkbox("Anaemia"):
                            filter1=1
                        else:
                            filter1=0

                        if st.checkbox("Diabetes"):
                            filter2=1
                        else:
                            filter2=0
                        if st.checkbox("high_blood_pressure"):
                            filter3=1
                        else:
                            filter3=0
                        if st.checkbox("Smoking"):
                            filter4="Smoking"
                        else:
                            filter4="Non Smoking"

                        idx =np.where((Heart['anaemia']==filter1) & (Heart['diabetes']==filter2) & (Heart['high_blood_pressure']==filter3) & (Heart['smoking']==filter4))
                        heart1=Heart.loc[idx]
                     #Plot the Graph
                        Affecting_Factors=heart1.groupby(['Age_Categ','sex'],as_index=False)['DEATH_EVENT'].sum()
                        fig = px.bar(Affecting_Factors, x="Age_Categ", y="DEATH_EVENT",color="sex",barmode="group",title="Number of Death Cases per Age Category",width=800, height=400)
                        fig.update_layout(
                        autosize=False,
                        yaxis=dict(
                        titlefont=dict(size=20))
                )

                        st.plotly_chart(fig)
                        if st.checkbox('Show Data'):
                            st.write(heart1)





                #Let's predict using the randomforest model_selectio



                def user_input_features():

                    st.write("""**1. Select Age :**""")
                    age = st.slider('', 0, 100, 25)
                    st.write("""**You selected this option **""",age)

                    st.write("""**2. Select Gender :**""")
                    sex = st.selectbox("(1=Male, 0=Female)",["1","0"])
                    st.write("""**You selected this option **""",sex)

                    st.write("""**3. anaemia :**""")
                    anaemia= st.selectbox("(1=Yes, 0=No)",["1","0"])
                    st.write("""**You selected this option **""",anaemia)

                    st.write("""**4. creatinine_phosphokinase :**""")
                    creatinine_phosphokinase = st.slider('In mm/Hg unit', 0, 8000, 100)
                    st.write("""**You selected this option **""",creatinine_phosphokinase)

                    st.write("""**5. diabetes :**""")
                    diabetes= st.selectbox("(1=yes, 0=no)",["1","0"])
                    st.write("""**You selected this option **""",diabetes)

                    st.write("""**6. ejection_fraction :**""")
                    ejection_fraction = st.slider('', 0, 100, 10)
                    st.write("""**You selected this option **""",ejection_fraction)

                    st.write("""**7. high_blood_pressure :**""")
                    high_blood_pressure = st.selectbox("(1=Y, 0=N)",["1","0"])
                    st.write("""**You selected this option **""",high_blood_pressure)

                    st.write("""**8.platelets :**""")
                    platelets = st.slider('', 0, 500000, 10000)
                    st.write("""**You selected this option **""",platelets)

                    st.write("""**9. serum_creatinine :**""")
                    serum_creatinine = float(st.slider('', 0.0, 5.0, 0.2))
                    st.write("""**You selected this option **""",serum_creatinine)

                    st.write("""**10. serum_sodium :**""")
                    serum_sodium = st.slider( '',0, 200, 10)
                    st.write("""**You selected this option **""",serum_sodium)

                    st.write("""**11. smoking :**""")
                    smoking= st.selectbox("(1=Smoking, 0=Non_Smoking)",["1","0"])
                    st.write("""**You selected this option **""",smoking)

                    st.write("""**12. time :**""")
                    time = st.slider( '',0, 300, 4)
                    st.write("""**You selected this option **""",time)

                    data = {'age': age, 'sex': sex, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase, 'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure, 'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,'smoking':smoking,'time':time}
                    features = pd.DataFrame(data, index=[0])
                    return features
                if select=="Prediction":
                    df = user_input_features()
                    st.subheader('Given Inputs : ')
                    st.write(df)


                    X = Heart1.iloc[:,0:12].values
                    Y = Heart1.iloc[:,[12]].values

                    model = RandomForestClassifier()
                    model.fit(X, Y)

                    prediction = model.predict(df)
                    st.subheader('Prediction :')
                    df1=pd.DataFrame(prediction,columns=['0'])
                    df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
                    df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
                    st.write(df1)

                    prediction_proba = model.predict_proba(df)
                    st.subheader('Prediction Probability in % :')
                    st.write(prediction_proba * 100)

        else:
                st.warning("Incorrect Username or Password")

    elif choice=="SignUp":
        st.subheader("Create New Account")
        new_username=st.sidebar.text_input("UserName ")
        new_password=st.sidebar.text_input("Password ",type='password')
        confirm_password=st.sidebar.text_input("Confirm Password ",type='password')
        if new_password==confirm_password:
            st.success("Confirmed Password")
        else:
            st.warning("Password not the same")
        if st.sidebar.button("Signup"):
            create_usertable()
            hashed_new_password=generate_hashes(new_password)
            add_userdata(new_username,hashed_new_password)
            st.success("You have created a new account")
            st.info("Go To Login Page")

main()


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
