from typing import Any
import os
import numpy as np
import boto3
import time
import random
import streamlit as st
import json
from streamlit.hello.utils import show_code

def _customize_css_style():
    # Note padding is top right bottom left
    st.markdown(
        """
        <style>
            div[data-testid=stToast] {
                padding: 20px 10px 40px 10px;
                width: 30%;
            }
             
        </style>
        """, unsafe_allow_html=True
    )

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
    
    if st.session_state.selected_models != None  :
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
        st.write("\n")
        st.write("##### 2. 클라이언트 초기화")
        code = f'### 클라이언트 선언\n'
        code += f'client = boto3.client(service_name="bedrock-runtime", region_name="{region}")\n\n'
        code += f'### 페이로드 준비\n'
        code += '프롬프트 = "Human: " + prompt + " Assistant:"\n'
        code += 'body = {\n'
        code += '    "prompt": 프롬프트,\n' 
        code += '    "max_tokens_to_sample": 200,\n'
        code += '    "temperature": 0.5,\n'
        code += '    "stop_sequences": ["Human:"],\n'
        code += '}'
        st.code(code, language="python", line_numbers=False)
        st.write("\n")
        st.write("##### 3. 응답 받아오기")
        code = f'### 모델 invoke\n'
        code += f'response = bedrock_runtime_client.invoke_model(\n'
        code += f'   modelId="{filtered_data["modelId"]}", body=json.dumps(body))\n\n'
        code += f'### 응답 요청\n'
        code += f'response_body = json.loads(response["body"].read())\n'
        code += f'answer = response_body["completion"]'
        st.code(code, language="python", line_numbers=False)

        chat_ui()

def toast(message):
    st.toast(message)

def bot_response(query):

    origin_text = '''
    enclosed_prompt = "Human: " + query + "\n\nAssistant:"

    client = boto3.client(service_name="bedrock-runtime", 
                      region_name="us-east-1",
                      aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                      aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

    body = {
            "prompt": enclosed_prompt,
            "max_tokens_to_sample": 200,
            "temperature": 0.5,
            "stop_sequences": ["\n\nHuman:"],
            }

    response = client.invoke_model(
        modelId="anthropic.claude-v2", body=json.dumps(body)
        )

    response_body = json.loads(response["body"].read())
    answer = response_body["completion"]
    '''
    toast(origin_text)


    enclosed_prompt = "Human: " + query + "\n\nAssistant:"

    client = boto3.client(service_name="bedrock-runtime", 
                      region_name="us-east-1",
                      aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                      aws_secret_access_key=os.environ['AWS_SECRET_KEY'])

    body = {
            "prompt": enclosed_prompt,
            "max_tokens_to_sample": 200,
            "temperature": 0.5,
            "stop_sequences": ["\n\nHuman:"],
            }

    response = client.invoke_model(
        modelId="anthropic.claude-v2", body=json.dumps(body)
        )
    toast(response)
    response_body = json.loads(response["body"].read())
    answer = response_body["completion"]

    


    with st.chat_message("assistant"):
        response = st.write(str(answer))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

def chat_ui():
    st.write("### Text Inference")
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
        bot_response(prompt)
    
    
st.set_page_config(page_title="BedRock Demo", page_icon="📹")
_customize_css_style()
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
st.write("##### 1. 선택된 모델정보")

model_desc = st.container()
col_m1, col_m2, col_m3 = model_desc.columns([1,0.5,0.5])
get_model_desc()

