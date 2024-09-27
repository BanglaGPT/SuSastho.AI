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

from flask import Flask, jsonify, request
import time
import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm.cli import tqdm
import numpy as np
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import random
from vectordb import load_model, embed_search, embed_search_batch, tiktoken_len
from dotenv import load_dotenv
import os, re
import requests
from joblib import Parallel, delayed
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

# load model
print('Model Loading')
tokenizer = AutoTokenizer.from_pretrained(os.environ['LLM_TOKENIZER'], use_auth_token=False)
tokenizer.padding_side = 'right'

model_emb, tokenizer_emb, context_docs, context_embeddings = load_model(
    data_path=os.environ['DATA_DOC_PATHS'],    #'./nlp_api/data/Fixed/*',
    chunk_size=288, chunk_overlap=0, 
    separators=["\n\n", "\n", "।", "|", "?", ";", "!", ",", "-", "*", " ", ""],
    device='cuda'
)


with open(os.environ['PROMPT_SEARCHQ'], 'r') as f:
    searchq_prompt = f.read()
with open(os.environ['PROMPT_QA'], 'r') as f:
    ans_prompt = f.read()
print('\nModel loaded\n\n\n')



def get_context_from_prompt(query:str, max_gpt_tokens:int, max_bloom_tokens:int, shuffle_chunks:bool):
    result = embed_search(
        query, 
        docs=context_docs,
        embeddings=context_embeddings,
        model_emb=model_emb,
        tokenizer_emb=tokenizer_emb,
        topn=40
    )
    
    filtered_result = []
    gpt_token_len = 0
    bloom_token_len = 0
    for i in result:
        l = tiktoken_len(i['docs'].page_content + i['docs'].metadata['question'])
        if l + gpt_token_len > max_gpt_tokens:
            break
        
        bl = len(tokenizer.encode(i['docs'].page_content + i['docs'].metadata['question']))
        if bl + bloom_token_len > max_bloom_tokens:
            break
        
        filtered_result.append(i)
        gpt_token_len += l
        bloom_token_len += bl
    
    if shuffle_chunks:
        random.shuffle(filtered_result)
    return filtered_result, gpt_token_len, bloom_token_len



def get_contexts_from_prompt_batch(query:list, max_gpt_tokens:int, max_bloom_tokens:int, min_threshold=0, max_threshold=100, topn=15):
    result = embed_search_batch(
        query, 
        docs=context_docs,
        embeddings=context_embeddings,
        model_emb=model_emb,
        tokenizer_emb=tokenizer_emb,
        topn=topn
    )
    result = [x for x in result if x['score'] > min_threshold and x['score'] < max_threshold]
    
    final_results = []
    
    filtered_result = []
    gpt_token_len = 0
    bloom_token_len = 0
    for i in result:
        gl = tiktoken_len(i['docs'].page_content + i['docs'].metadata['question'])
        if gl + gpt_token_len > max_gpt_tokens:
            final_results.append(filtered_result)
            
            # Create new sample
            filtered_result = []
            gpt_token_len = 0
            bloom_token_len = 0
        
        bl = len(tokenizer.encode(i['docs'].page_content + i['docs'].metadata['question']))
        if bl + bloom_token_len > max_bloom_tokens:
            final_results.append(filtered_result)
            
            # Create new sample
            filtered_result = []
            gpt_token_len = 0
            bloom_token_len = 0
        
        filtered_result.append(i)
        gpt_token_len += gl
        bloom_token_len += bl
    
    return final_results



def generate(text, temperature, max_new_tokens, stop=None):
    headers = {'Content-Type': 'application/json', 'authorization': os.environ['FIREWORKS_API_KEY']}
    data = {
        "model": os.environ['FIREWORKS_MODEL_PATH'], #"./nlp_api/model_llama3/checkpoint-360-marged",
        "prompt": text,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    
    if stop:
        data['stop'] = stop
    
    resp = requests.post(os.environ['FIREWORKS_ENDPOINT'], json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json()['choices'][0]['text']
    
    return None



def GenSearchQuery(query, query_ctx, temparature, max_new_tokens):
    input_text = searchq_prompt.replace('<history>', query_ctx).replace('<query>', query)
    print('>>>> SearchQ Input: ', input_text)
    
    response = generate(
        input_text, temperature=temparature, max_new_tokens=max_new_tokens,
    )
    print('>>>> SearchQ Response: ', response)
    
    statequery = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=1), response)[0]
    if 'Not Required' in response:
        print('>>>> Search Query not required')
        return statequery, [[]]
    
    response_tmp = response + '\n\nStep'
    q_en = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=3), response_tmp)[0].replace('-','').replace('"','').strip()
    q_en = q_en.split('\n')
    q_en = [x.strip() for x in q_en]
    #q_bn = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=3), response)[0].replace('-','').replace('"','').strip()
    
    print('>>>> Search: ', statequery, q_en)
    return statequery, [q_en]
    


def GetScores(query, query_ctx, statequery, context, temparature, max_new_tokens):
    input_text = ans_prompt.replace('<history>', query_ctx).replace('<query>', query).replace('<context>', context).replace('<statequery>', statequery)
    print('>>>> QA Input: ', input_text)
    
    response = generate(
        input_text, temperature=temparature, max_new_tokens=max_new_tokens, stop='Step 6'
    )
    print('>>>> QA Response: ', response)
    
    response_tmp = response + '\n\nStep'
    score = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=5), response_tmp)[0]
    
    score = score.replace('-', '').strip()
    score = score.split(' ')[0]
    score = float(score)
    
    print('>>>> Score: ', score)
    return score, input_text



def GetAnswer(input_text, temparature, max_new_tokens):
    response = generate(
        input_text, temperature=temparature, max_new_tokens=max_new_tokens,
    )
    print('>>>> QA Response: ', response)
    
    response_tmp = response + '\n\nStep'
    answer = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=7), response_tmp)[0]
    answer = re.sub(r'^-','',answer.strip()).strip()# answer.replace('-','').strip()
    
    print('>>>> Answer: ', answer)
    return answer



not_found_response = "দুঃখিত আমি আপনাকে এ ব্যাপারে সাহায্য করতে পারছি না।"
max_bloom_tokens = 1728
max_gpt_tokens = 2900

def chat_completion(dialog, model_name, mode_name, msg_context_size, ctx_checker_tmp, lm_tmp, topn, max_ctx, cls_threshold, llm_enable):
    log = {
        'retrived_docs': [], 'retrived_prob':[], 'matched_docs': [], 'matched_prob':[], 
        'llm_input': '', 'llm_reasoning':'', 'llm_response': '', 'module': ''
    }
    
    # Create message and message context
    query = dialog[-1]['content'].strip()
    msg_ctx = '\n'.join(x['content'] for x in dialog[-msg_context_size:-1])
    msg_ctx = 'None' if len(msg_ctx.strip()) == 0 else msg_ctx
    log['message_ctx'] = msg_ctx
    print('>>> msg_ctx: ', msg_ctx)
    
    
    # TODO Try multiple time if fails
    statequery, search_str = GenSearchQuery(
        query=query,
        query_ctx=msg_ctx,
        temparature=0.04,
        max_new_tokens=600,
    )
    
    # Only take english strings
    search_str = search_str[0]
    if len(search_str) == 0:
        print('>>>> No search string was generated')
        return not_found_response, log
    
    
    # Returns group of docs
    docs = get_contexts_from_prompt_batch(
        query=search_str, 
        max_gpt_tokens=max_gpt_tokens, 
        max_bloom_tokens=max_bloom_tokens,
        min_threshold=78,
        max_threshold=100,
        topn=10,
    )
    docs = docs[:3]   # Only try 4 times
    
    def parallel_score(doc):
        context = '\n---\n'.join([f"FAQ: {x['docs'].metadata['question']}\nAnswer: {x['docs'].page_content}" for x in doc])
        score, input_text = GetScores(
            query=query,
            query_ctx=msg_ctx,
            statequery=statequery,
            context=context,
            temparature=0,
            max_new_tokens=1200,
        )
        return {'score': score, 'input_text': input_text, 'docs': doc}
    responses = Parallel(n_jobs=3, backend="threading")(delayed(parallel_score)(doc) for doc in docs)
    
    
    responses = sorted(responses, key=lambda x: x['score'], reverse=True)
    print('>>>> Responses:', responses)
    
    if len(responses) == 0 or responses[0]['score'] < 1.6:
        return not_found_response, log
    
    
    # Get best answer
    fanswer_dict = responses[0]
    # Generate best answer
    fanswer = GetAnswer(
        input_text=fanswer_dict['input_text'],
        temparature=0,
        max_new_tokens=1200,
    )
    ans_docs = fanswer_dict['docs']
    # Log
    # TODO Input of Search Query
    # TODO Input of QA
    log['retrived_prob'] = [x['score'] for x in ans_docs]
    log['retrived_docs'] = [f"FAQ: {x['docs'].metadata['question']}\nAnswer: {x['docs'].page_content}" for x in ans_docs]
    log['matched_docs'] = log['retrived_docs']
    log['matched_prob'] = log['retrived_prob']
    #log['llm_reasoning'] = response
    log['llm_response'] = fanswer
    return fanswer, log



@app.route('/')
def index():
    return "NLP Backend API"



@app.route('/llm/v1/api', methods = ['POST'])
def llm_infer():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        
        json['chat_history'] = [x for x in json['chat_history'] if x['role']=='user']
        
        ## Need to validate json
        # need to chat history. must not be empty
        
        cls_threshold = 0.33
        ctx_checker_tmp = 0.008
        lm_tmp = 0.12
        max_ctx = 3
        llm_enable = True
        if 'config' in json:
            ctx_checker_tmp = json['config']['ctx_checker_tmp']
            lm_tmp = json['config']['lm_tmp']
            max_ctx = json['config']['max_ctx']
            
            cls_threshold = json['config']['cls_threshold']
            llm_enable = json['config']['llm_enable']
            
            print('Debug Mode:  ', ctx_checker_tmp, lm_tmp)
        
        
        ## Run Model
        print('Transcription Started')
        resp, log = chat_completion(
            json['chat_history'], 
            model_name=json['model'], 
            mode_name=json['mode'],
            msg_context_size=5,   # -1 is the actual message context size
            ctx_checker_tmp=ctx_checker_tmp,
            lm_tmp=lm_tmp,
            topn = 24,
            max_ctx=max_ctx,
            cls_threshold=cls_threshold,
            llm_enable = llm_enable,
        )
        print(resp)
        
        data = {
            'responses': [
                {'role': 'ai', 'content': resp, 'meta': log['module']}
            ],
            'refer_needed': False,
            'logs': {
                'version': '1',
                'content': {
                    'context_llm': {'generated_ctx': ''},
                    'retrival_model': {
                        'retrived_doc': log['retrived_docs'][:16],
                        'retrived_prob': [float(x) for x in log['retrived_prob']][:16],
                        'matched_doc': log['matched_docs'],
                        'matched_prob': [float(x) for x in log['matched_prob']]
                    },
                    'llm': {
                        'response': log['llm_response'],
                        'reasoning': log['llm_reasoning'],
                        'input': log['llm_input'],
                        'message_ctx': log['message_ctx']
                    },
                    'risk_llm': {
                        
                    },
                    'risk_detection': {
                        
                    },
                    'sop_module': {
                        
                    },  
                }
            }
        }
        return jsonify({'data': data})
    return 'Content-Type not supported!', 400



    
  
# driver function
if __name__ == '__main__':
    app.run(debug=False, port=5000)
