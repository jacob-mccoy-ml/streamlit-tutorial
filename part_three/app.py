import streamlit as st
import pandas as pd
from resources import get_train_data, clean_data, train_model

train_data = get_train_data()

edited = st.data_editor(train_data)

cleaned_data = clean_data(train_data)

edited_2 = st.data_editor(cleaned_data)

model = train_model(edited_2)

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

train_data = get_train_data()

edited = st.data_editor(train_data)

cleaned_data = clean_data(train_data)

edited_2 = st.data_editor(cleaned_data)

model = train_model(edited_2)

if model:
    st.write('Model successfully trained:')

fake_data = pd.DataFrame(data=[[3,1,22,1,0,7.25,2]], columns=cleaned_data.drop('Survived', axis=1).columns.tolist())

predicted_label = model.predict(fake_data)
predicted_proba = model.predict_proba(fake_data)

st.write(predicted_label)
st.write(predicted_proba)
"""

st.code(code, language="python", line_numbers=True)