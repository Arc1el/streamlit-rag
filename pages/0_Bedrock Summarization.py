from typing import Any
import os
import numpy as np
import boto3

import streamlit as st
from streamlit.hello.utils import show_code

def get_bedrock_models(region):
    aws_access_key_id = os.environ['AWS_ACCESS_KEY']
    aws_secret_access_key = os.environ['AWS_SECRET_KEY']

    client = boto3.client(
        'bedrock',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region  # 예시로 사용된 리전, 실제 사용 리전에 맞게 변경하세요
    )

    # 기반 모델 리스트를 가져오는 요청
    model_list = []
    
    response = client.list_foundation_models(byOutputModality='TEXT')
    datas = response["modelSummaries"]
    for data in datas :
        model_list.append(data["modelId"])
    
    st.session_state.origin_data = datas

    providers = set()


    for data in datas:
        model_str = data["modelId"]
        provider, model = model_str.split('.', 1)
        providers.add(provider)

    providers = list(providers)

    selected_provider = col2.selectbox("프로바이더를 선택하세요", providers, index=None,)

    models = []

    for data in datas:
        model_str = data["modelId"]
        provider, model = model_str.split('.', 1)
        
        if selected_provider == provider:
            models.append(model)

    selected_models = col3.selectbox("모델을 선택하세요", models, index=None)

    st.session_state.selected_provider = selected_provider
    st.session_state.selected_models = selected_models


def get_model_desc():

    model_str = str(st.session_state.selected_provider) + "." + str(st.session_state.selected_models)
    origin_data = st.session_state.origin_data
    
    filtered_data = [model for model in origin_data if model["modelId"] == model_str][0]

    model_name = filtered_data["modelName"]
    provider_name = filtered_data["providerName"]
    model_arn = filtered_data["modelArn"]

    col_m1.write("###### `model_arn`")
    col_m1.write(model_arn)
    col_m2.write("###### `model_name`")
    col_m2.write(model_name)
    col_m3.write("###### `provider_name`")
    col_m3.write(provider_name)

def response_generator():
    response = random.choice(
        [
            "안녕하세요",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    
st.set_page_config(page_title="BedRock Demo", page_icon="📹")
st.markdown("# Bedrock Text Demo")
config = st.container()
col1, col2, col3 = config.columns([1,1,1])


regions = ["us-east-1", "us-west-2", "ap-southeast-1", "ap-northeast-1", "eu-central-1"]

region = col1.selectbox(
    '리전을 선택하세요',
    regions,
    index=1,)

get_bedrock_models(region)
st.write("")
st.write("##### 선택된 모델정보")

model_desc = st.container()
col_m1, col_m2, col_m3 = model_desc.columns([1,0.5,0.5])
get_model_desc()

st.write("### Text Inference")


import random
import time
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

with st.chat_message("assistant"):
    response = st.write_stream(response_generator())
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})