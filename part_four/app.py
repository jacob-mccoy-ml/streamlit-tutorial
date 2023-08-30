import streamlit as st
import pandas as pd
from resources import get_train_data, clean_data, train_model
from datetime import timedelta

st.set_page_config(
    page_title="Titanic Survivability Model", page_icon="🚢", layout="wide"
)
col_1, col_2 = st.columns([2,2])

@st.cache_data(ttl=timedelta(hours=24))
def datawork():
    train_data = get_train_data()
    cleaned_data = clean_data(train_data)
    model = train_model(cleaned_data)

    return model, cleaned_data, train_data

model, cleaned_data, train_data = datawork()

class_num = col_1.slider("Select Class:", 1, 3)
sex = col_1.selectbox("Select Sex:", ['male', 'female'])
age = col_1.slider("Select Age:", 0, 80)
sibsp = col_2.slider("Siblings and Spouses:", 0, 8)
parch = col_2.slider("Parents and Children:", 0, 6)
fare = col_2.slider("Fare:", 0, 515, step=5)

if model:
    st.write('Model successfully trained:')

fake_data = pd.DataFrame(data=[[3,1,22,1,0,7.25,2]], columns=cleaned_data.drop('Survived', axis=1).columns.tolist())

predicted_label = model.predict(fake_data)
predicted_proba = model.predict_proba(fake_data)

st.write(predicted_label)
st.write(predicted_proba)


code ="""
import streamlit as st
import pandas as pd
from resources import get_train_data, clean_data, train_model
from datetime import timedelta
from time import sleep

st.set_page_config(
    page_title="Titanic Survivability Model", page_icon="🚢", layout="wide"
)
col_1, col_2 = st.columns([2,2])

@st.cache_data(ttl=timedelta(hours=24))
def datawork():
    train_data = get_train_data()
    cleaned_data = clean_data(train_data)
    model = train_model(cleaned_data)

    return model, cleaned_data, train_data

model, cleaned_data, train_data = datawork()

class_num = col_1.slider("Select Class:", 1, 3)
sex = col_1.selectbox("Select Sex:", ['male', 'female'])
age = col_1.slider("Select Age:", 0, 80)
sibsp = col_2.slider("Siblings and Spouses:", 0, 8)
parch = col_2.slider("Parents and Children:", 0, 6)
fare = col_2.slider("Fare:", 0, 515, step=5)

if model:
    st.write('Model successfully trained:')

fake_data = pd.DataFrame(data=[[3,1,22,1,0,7.25,2]], columns=cleaned_data.drop('Survived', axis=1).columns.tolist())

predicted_label = model.predict(fake_data)
predicted_proba = model.predict_proba(fake_data)

st.write(predicted_label)
st.write(predicted_proba)
"""

st.code(code, language="python", line_numbers=True)