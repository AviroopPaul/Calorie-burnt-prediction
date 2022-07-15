import pickle
import streamlit as st
import numpy as np

calorie_model=pickle.load(open('D:/ML-projects/Calorie-burnt-prediction/calorie_burnt.sav','rb'))

st.title("Calorie Burnt Prediction using Machine Learning")

Gender=st.text_input("Gender")
Age=st.text_input("Age")
Height=st.text_input("Height")
Weight=st.text_input("Weight")
Duration=st.text_input("Duration")
HeartRate=st.text_input("Heart Rate")
BodyTemperature=st.text_input("Body Temperature")

caloriesBurnt=0

inputData=(Gender,Age,Height,Weight,Duration, HeartRate, BodyTemperature)
numpyArray=np.array(inputData,dtype=float)
reshapedArray=numpyArray.reshape(1,-1)


if(st.button('Find Calories Burnt')):
    caloriesBurnt=calorie_model.predict(reshapedArray)
st.write('Calories Burnt(in kcal): ')
caloriesBurnt = "{:.2f}".format(caloriesBurnt[0])
st.success(caloriesBurnt) 
# st.write('kcal')


