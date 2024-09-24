import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score
import streamlit as st 
import joblib

st.title(':orange[Есть ли сердечное заболевание?]  :heart:')
st.subheader('***:violet[Больны ли вы?]***')


ml_pipeline = joblib.load('ml_pipeline_CB_ml.pkl')

with st.form(key='my_form'):
    Age = st.text_input(label='Укажите ваш возраст:')
    Sex = st.text_input(label='Укажите ваш пол (M/F):')
    ChestPainType = st.text_input(label='Укажите тип боли в груди (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic):')
    RestingB = st.text_input(label='Укажите кровяное давление:')
    Cholesterol = st.text_input(label='Укажите уровень холестерина:')
    FastingBS = st.text_input(label='Укажите уровень сахара (1 если больше 120 mg/dl в ином случае 0)')
    RestingECG = st.text_input(label='Укажите ваш ЭКГ (Normal, ST, ST-T):')
    MaxHR = st.text_input(label='Укажите вашу максимальную ЧСС:')
    ExerciseAngina = st.text_input(label='Есть ли у вас стенокардия? (Y/N):')
    Oldpeak = st.text_input(label='Укажите ваш пиковый низкий?:')
    ST_Slope = st.text_input(label='Укажите вашу максимальную нагрузку ST (Flat/Up/Down):')   
    submit_button = st.form_submit_button(label='Отправить')   

    
data = {
'Age': [Age],
'Sex': [Sex],
'ChestPainType': [ChestPainType],
'RestingBP': [RestingB],
'Cholesterol': [Cholesterol],
'FastingBS': [FastingBS],
'RestingECG': [RestingECG],
'MaxHR': [MaxHR],
'ExerciseAngina': [ExerciseAngina],
'Oldpeak': [Oldpeak],
'ST_Slope': [ST_Slope]
}

input = pd.DataFrame(data)
try:
    if ml_pipeline.predict(input)==1:
        st.subheader('Ваше сердце болит!  :disappointed:')
    else:
        st.subheader('Поздравляю, вы здоровы!  :smile:')
except:
    st.stop()
