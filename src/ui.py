# ----------------------------------------------------------------------------------
# Author: Shamim Ahamed
# Email: shamim.aiexpert@gmail.com
# Affiliation: Advanced Intelligent Multidisciplinary Systems Lab(AIMS Lab), 
#              Institute of Research Innovation, Incubation and Commercialization(IRIIC), 
#              United International University, Dhaka 1212, Bangladesh
# Description: This Python script is developed as part of ongoing research at AIMS Lab and IRIIC.
#              It is intended for academic and research purposes only.
# Date: 24 September 2024
# ----------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from tqdm.cli import tqdm
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm


def get_user_data(api, parameters):
    response = requests.post(f"{api}", json=parameters)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"ERROR: {response.status_code}")
        return None

st.set_page_config(page_title="SuSastho.AI Chatbot", page_icon="🚀")

st.markdown("""
    <style>
        p {
            font-size:0.95rem !important;
        }
        textarea {
            font-size: 0.95rem !important;
            padding: 0.8rem 1rem 0.75rem 0.8rem !important;
        }
        
        footer {visibility: hidden;}
        #MainMenu, header {visibility: hidden;}
        
        /*User*/
        .st-emotion-cache-1c7y2kd {
            background-color: rgb(237 241 245 / 90%);
            padding: 1rem 0.6rem 1rem 1rem;
        }
        
        /*AI*/
        .st-emotion-cache-4oy321 {
            background-color: rgb(213 229 235 / 82%);
            padding: 1rem 0.6rem 1rem 1rem;
        }
        
        
        /*AI avatar background*/
        .st-emotion-cache-1lr5yb2{
            background-color: rgb(65 173 203);
        }
        
        
            
        /*Fix Chat send button padding*/
        .stChatFloatingInputContainer button{
            padding: 0.76rem 0.62rem !important;
        }
            
        
        hr {
            margin: 0em 0px 2em 0em !important;
        }
        
        body {
            zoom: 0.9;
        }
    </style>
""", unsafe_allow_html=True)
    

st.markdown('### SuSastho.AI')
st.markdown('-----')

endpoint = 'http://127.0.0.1:5000/llm/v1/api'

def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": 'assistant', "content": 'হ্যালো! আমি একটি এআই অ্যাসিস্ট্যান্ট। কীভাবে সাহায্য করতে পারি? 😊'}]
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("এখানে মেসেজ লিখুন"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        
        ## Get context
        params = {
            "chat_history": [{"content": x["content"], "role": x["role"]} for x in st.session_state.messages[-10:] if x['role']=='user'],
            "model": "bloom-7b",
            "mode": "specific"
        }
        resp = get_user_data(endpoint, params)
        if resp == None:
            st.markdown('#### INTERNAL ERROR')
            return

        print(resp['data']['logs']['content'])
        response = resp['data']['responses'][0]['content']
        reasoning = resp['data']['logs']['content']['llm']['reasoning']
        llm_input = resp['data']['logs']['content']['llm']['input']
        context = resp['data']['logs']['content']['retrival_model']['matched_doc']
        context_prob = resp['data']['logs']['content']['retrival_model']['matched_prob']
        
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar=None):
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
if __name__ == '__main__':
    main()
