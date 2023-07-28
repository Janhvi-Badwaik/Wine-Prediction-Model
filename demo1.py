import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import mysql.connector
import streamlit as st
from PIL import Image


#connection
mydb=mysql.connector.connect(host="localhost",user="root",password="rahel@777",database="demo")
mycursor=mydb.cursor()  


# Prediction model
def modelWine():
    #Input of Components
    def user_input_features():
        input_alcohol = st.number_input('Alcohol : ')
        input_Malic_Acid = st.number_input('Malic Acid : ')
        input_Ash = st.number_input('Ash : ')
        input_Ash_Alcanity = st.number_input('Ash Alcanity : ')
        input_Magnesium = st.number_input('Magnesium : ')
        input_Total_Phenols = st.number_input('Total Phenol : ')
        input_Flavanoids = st.number_input('Flavanoids : ')
        input_Nonflavanoids_Phenols = st.number_input('Non Flavanoid Phenols : ')
        input_Proanthocyanins = st.number_input('Proanthocyanins : ')
        input_Color_Intensity = st.number_input('Color Intensity : ')
        input_Hue = st.number_input('Hue : ')
        input_OD280 = st.number_input('OD280 : ')
        input_Proline = st.number_input('Proline : ')

        lst={'Alcohol' : input_alcohol,
            'Malic_Acid' : input_Malic_Acid,
            'Ash' : input_Ash,
            'Ash_Alcanity' : input_Ash_Alcanity,
            'Magnesium' : input_Magnesium,
            'Total_Phenol' : input_Total_Phenols,
            'Flavanoids' : input_Flavanoids,
            'Nonflavanoid_Phenols' : input_Nonflavanoids_Phenols,
            'Proanthocyanins' :input_Proanthocyanins,
            'Color_Intensity' : input_Color_Intensity,
            'Hue' : input_Hue,
            'OD280' : input_OD280,
            'Proline' : input_Proline}

        features = pd.DataFrame(lst, index=[0])
        return features
    
    data = pd.read_csv('wine.csv')
    data.head()
    data.isnull().sum()
    
    #Dependent and independent variable sepration
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    
    #Splitting train and test data 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    #Dimensionality reduction using LDA
    lda = LDA(n_components = 2)
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)
    
    #Traing Model using RandomForestClassifier
    classifier = RandomForestClassifier(random_state = 0)
    classifier.fit(x_train, y_train)
    
    #Testing the test data
    y_pred = classifier.predict(x_test)
    
    #Checking the Accuracy of the model
    print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
    
    #Frontend
    st.title("Predicting Customer segment for Wine :bar_chart: :chart_with_upwards_trend:🍷 ")
    st.write(" ")
    st.write("**Note : Enter the quantity of constituents in the New wine**")
    st.write(" ")
    st.write(" ")
    df = user_input_features()
    st.subheader('User Input parameters')
    st.write(df)
    input_data_lda = lda.transform(df)
    predict_btn = st.button("Predict")
    if predict_btn:  
        prediction = classifier.predict(input_data_lda)

        st.subheader('Customer Segment')
        st.write(prediction)



def main():
    
    
    menu = ["Home","Login","SignUp","LogOut"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    #Home Page
    if choice == "Home":
        st.title("WELCOME to WineSpectrum")
        st.write(" ")
        st.markdown("Discover the perfect wine for every occasion with our innovative app powered by advanced dimensionality reduction using LDA, bringing you the most refined and personalized customer segment.")
        st.write("  ")
        image1=Image.open("C:/Users/rahel/OneDrive/Desktop/Python_pr/konverge/bottles.jpg")
        st.image(image1)
        
    #Login Page
    elif choice == "Login": 
        lusername = st.sidebar.text_input("**Username**")
        lpassword = st.sidebar.text_input("**Password**",type='password')
        if st.sidebar.checkbox("Login"):
            sql="select opassword from account where username = %s and opassword = %s"
            val=(lusername,lpassword)
            mycursor.execute(sql,val)
            data=mycursor.fetchone()
            mydb.commit()
            if  data:
                st.sidebar.success("Logged in {}".format(lusername))
                modelWine()                
            else:
                st.sidebar.warning("Incorrect Username/Password")
     #LogOut Page           
    elif choice == "LogOut":
        st.header("You have Logged Out Successfully!")
        st.header("THANK YOU 🥂")
       
    #SignUp Page    
    elif choice == "SignUp":
        st.subheader("**Create new Account**")
        email = st.text_input("**Email**")
        username = st.text_input("**Username**")
        opassword = st.text_input("**Password**",type='password')
        cpassword = st.text_input("**Confirm Password**",type='password')
        try:
            if st.button("SignUp"):
                if opassword==cpassword:
                    sql = "insert into account(email,username,opassword,cpassword) values(%s,%s,%s,%s)"
                    val=(email,username,opassword,cpassword)
                    mycursor.execute(sql,val)
                    mydb.commit()
                    st.success("You have successfully created a valid Account")
                    st.info("Go to Login Menu to Login")
                else:
                    st.warning("Confirm password is not same")
        except:
            st.warning("Account is already created")
    
    
    
    
    
if __name__ == '__main__':
    main()