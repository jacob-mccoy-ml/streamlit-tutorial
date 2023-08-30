import streamlit as st

username = st.text_input("Enter username:")

if username:
    st.markdown(f"Hello {username}!")
else:
    st.markdown("Enter username above to see the output!")

code ="""
import streamlit as st

username = st.text_input("Enter username:")

if username:
    st.markdown(f"Hello {username}!")
else:
    st.markdown("Enter username above to see the output!")
"""

st.code(code, language="python", line_numbers=True)