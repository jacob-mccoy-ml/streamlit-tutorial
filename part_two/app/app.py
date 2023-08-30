import streamlit as st
import pandas as pd

breakfast_choices = st.multiselect("Choose breakfast foods:", options=['eggs', 'bacon', 'pancakes', 'cereal', 'toast', 'coffee'])

st.markdown(f"My ideal breakfast consists of {breakfast_choices}")

df_dictionary = {
    'col1': [1,2,3],
    'col2': [4,5,6]
}

df = pd.DataFrame(data=df_dictionary)

edited = st.data_editor(df)

col1_sum = edited['col1'].sum()
col2_sum = edited['col2'].sum()
st.markdown(f"Column one sums to: {col1_sum}; Column two sums to: {col2_sum}")

code ="""
import streamlit as st
import pandas as pd

breakfast_choices = st.multiselect("Choose breakfast foods:", options=['eggs', 'bacon', 'pancakes', 'cereal', 'toast', 'coffee'])

st.markdown(f"My ideal breakfast consists of {breakfast_choices}")

df_dictionary = {
    'col1': [1,2,3],
    'col2': [4,5,6]
}

df = pd.DataFrame(data=df_dictionary)

edited = st.data_editor(df)

col1_sum = edited['col1'].sum()
col2_sum = edited['col2'].sum()
st.markdown(f"Column one sums to: {col1_sum}; Column two sums to: {col2_sum}")
"""

st.code(code, language="python", line_numbers=True)