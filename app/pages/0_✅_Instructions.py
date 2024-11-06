import sys
import os
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../../')))

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("INSTRUCTIONS.md").read())
