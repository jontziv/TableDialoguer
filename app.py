import streamlit as st
import pandas as pd
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import os
from pandasai import SmartDataframe # SmartDataframe for interacting with data using LLM
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd

st.title("Welcome to TableDialoguer - A chatbot to query your tabular data")

load_dotenv() 

groq_api_key = os.environ['GROQ_API_KEY']
st.session_state.df = None
st.session_state.prompt_history = []
if st.session_state.df is None:
    uploaded_file = st.file_uploader(
        "Choose a CSV file. This should be in long format (one datapoint per row).",
        type=["csv","xlsx"],
    )
    if uploaded_file is not None:
        format=Path(uploaded_file.name).suffix
        if format==".csv":
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
        else :
            data_xls = pd.read_excel(uploaded_file, 'Sheet1', dtype=str, index_col=None, engine="xlrd")
            uploaded_file+=data_xls.to_csv(uploaded_file, encoding='utf-8', index=False)
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df

    with st.form("Question"):
        question = st.text_input("Question", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner():
                llm = ChatGroq(
                groq_api_key=groq_api_key, model_name="llama3-70b-8192",
                            temperature=0.2)

    # Initialize SmartDataframe with DataFrame and LLM configuration
                pandas_ai = SmartDataframe(df, config={"llm": llm})
                x = pandas_ai.chat(question)

                if os.path.isfile('temp_chart.png'):
                    im = plt.imread('temp_chart.png')
                    st.image(im)
                    os.remove('temp_chart.png')

                if x is not None:
                    st.write(x)
                st.session_state.prompt_history.append(question)

    if st.session_state.df is not None:
        st.subheader("Current dataframe:")
        st.write(st.session_state.df)

    st.subheader("Prompt history:")
    st.write(st.session_state.prompt_history)


if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None
