import streamlit as st
import pandas as pd
from resources import get_train_data, clean_data, train_model, INPUT_CONVERSION
from datetime import timedelta

st.set_page_config(
    page_title="Titanic Survivability Model", page_icon="🚢", layout="wide"
)
st.markdown("<h2 style='text-align: center;'>🚢 Titanic Survivability Model</h2>", unsafe_allow_html=True)
col_0, col_1, col_2, col_3 = st.columns([1,4,4,1])

@st.cache_data(ttl=timedelta(hours=24))
def datawork():
    train_data = get_train_data()
    cleaned_data = clean_data(train_data)
    model = train_model(cleaned_data)
    columns = cleaned_data.drop('Survived', axis=1).columns.tolist()

    return model, columns

def configure_class_num():
    class_num = col_1.slider("Select Class:", 1, 3)
    return class_num

def configure_sex():
    sex = col_1.selectbox("Select Sex:", ['male', 'female'])
    return sex

def configure_age():
    age = col_1.slider("Select Age:", 0, 80)
    return age

def configure_sibsp():
    sibsp = col_2.slider("Siblings and Spouses:", 0, 8)
    return sibsp

def configure_parch():
    parch = col_2.slider("Parents and Children:", 0, 6)
    return parch

def configure_fare():
    fare = col_2.slider("Fare:", 0, 515, step=5)
    return fare

def configure_embarked(column):
    embarked = column.selectbox('Select Embarking Port:', ['Southampton', 'Queenstown', 'Cherbourg'])
    return embarked

def build_prediction_input(class_num, sex, age, sibsp, parch, fare, embarked, columns):
    prediction_input = pd.DataFrame(data=[[class_num,INPUT_CONVERSION['sex'][sex],age,sibsp,parch,fare,INPUT_CONVERSION['embarked'][embarked]]], columns=columns)
    return prediction_input

def make_prediction(model, prediction_input):
    predicted_proba = model.predict_proba(prediction_input)
    survival_chance = predicted_proba[0][1] * 100
    survival_chance = round(survival_chance,1)
    return survival_chance

def main():
    model, columns = datawork()
    class_num = configure_class_num()
    sex = configure_sex()
    age = configure_age()
    sibsp = configure_sibsp()
    parch = configure_parch()
    fare = configure_fare()
    _, col_1, _ = st.columns([1,8,1])
    embarked = configure_embarked(col_1)
    prediction_input = build_prediction_input(class_num, sex, age, sibsp, parch, fare, embarked, columns)
    survival_chance = make_prediction(model, prediction_input)
    col_1.divider()
    st.markdown(f"<h2 style='text-align: center;'>Chance of Survival: {survival_chance}%</h2>", unsafe_allow_html=True)

main()

st.divider()

code ="""
import streamlit as st
import pandas as pd
from resources import get_train_data, clean_data, train_model, INPUT_CONVERSION
from datetime import timedelta

st.set_page_config(
    page_title="Titanic Survivability Model", page_icon="🚢", layout="wide"
)
st.markdown("<h2 style='text-align: center;'>🚢 Titanic Survivability Model</h2>", unsafe_allow_html=True)
col_0, col_1, col_2, col_3 = st.columns([1,4,4,1])

@st.cache_data(ttl=timedelta(hours=24))
def datawork():
    train_data = get_train_data()
    cleaned_data = clean_data(train_data)
    model = train_model(cleaned_data)
    columns = cleaned_data.drop('Survived', axis=1).columns.tolist()

    return model, columns

def configure_class_num():
    class_num = col_1.slider("Select Class:", 1, 3)
    return class_num

def configure_sex():
    sex = col_1.selectbox("Select Sex:", ['male', 'female'])
    return sex

def configure_age():
    age = col_1.slider("Select Age:", 0, 80)
    return age

def configure_sibsp():
    sibsp = col_2.slider("Siblings and Spouses:", 0, 8)
    return sibsp

def configure_parch():
    parch = col_2.slider("Parents and Children:", 0, 6)
    return parch

def configure_fare():
    fare = col_2.slider("Fare:", 0, 515, step=5)
    return fare

def configure_embarked(column):
    embarked = column.selectbox('Select Embarking Port:', ['Southampton', 'Queenstown', 'Cherbourg'])
    return embarked

def build_prediction_input(class_num, sex, age, sibsp, parch, fare, embarked, columns):
    prediction_input = pd.DataFrame(data=[[class_num,INPUT_CONVERSION['sex'][sex],age,sibsp,parch,fare,INPUT_CONVERSION['embarked'][embarked]]], columns=columns)
    return prediction_input

def make_prediction(model, prediction_input):
    predicted_proba = model.predict_proba(prediction_input)
    survival_chance = predicted_proba[0][1] * 100
    survival_chance = round(survival_chance,1)
    return survival_chance

def main():
    model, columns = datawork()
    class_num = configure_class_num()
    sex = configure_sex()
    age = configure_age()
    sibsp = configure_sibsp()
    parch = configure_parch()
    fare = configure_fare()
    _, col_1, _ = st.columns([1,8,1])
    embarked = configure_embarked(col_1)
    prediction_input = build_prediction_input(class_num, sex, age, sibsp, parch, fare, embarked, columns)
    survival_chance = make_prediction(model, prediction_input)
    col_1.divider()
    st.markdown(f"<h2 style='text-align: center;'>Chance of Survival: {survival_chance}%</h2>", unsafe_allow_html=True)

main()
"""

st.code(code, language="python", line_numbers=True)