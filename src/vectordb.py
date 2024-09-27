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

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import numpy as np
from glob import glob
import tiktoken
import docx2txt
import pandas as pd
import os
import textdistance
import pickle


DOCS_CACHE_PATH = './.cache/docs.pickle'
SIMILARITY_THRESHOLD = 0.85


tik_tokenizer = tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text):
    tokens = tik_tokenizer.encode(text,disallowed_special=())
    return len(tokens)


e5_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
def e5_len(text):
    tokens = e5_tokenizer.encode(text)
    return len(tokens)


def parse_doc(text, fname):
    rows = text.split('ID: ')
    
    docs = []
    metas = []
    for r in rows:
        arr = r.split('Context:')
        if len(arr) < 2:
            #print(r)
            continue
        
        id = int(arr[0].strip())
        d = arr[1].strip()
        
        if len(d) == 0:
            continue
        
        arr = d.split('Question:')
        ctx = arr[0].strip()
        d = arr[1].strip()
        
        arr = d.split('Answer:')
        q = arr[0].strip()
        a = arr[1].strip()
        
        docs.append(a)
        metas.append({'fname': fname, 'id': id, 'context': ctx, 'question': q})
        
        if id == 278:
            print(q)
            print(a)
        
    return docs, metas




def get_docs(base_path, chunk_size, chunk_overlap, separators):
    docs, metas = [], []
    
    docs_list = glob(base_path)
    for p in docs_list:
        #dtext = getText(p)
        extention = os.path.splitext(p)[1]
        if extention == '.docx':
            dtext = docx2txt.process(p)
        elif extention == '.txt':
            with open(p, encoding="utf8") as f:
                dtext = f.read()
        else:
            raise Exception(f'{extention} File type Not supported')
        
        d, m = parse_doc(dtext, os.path.basename(p))
        docs.extend(d)
        metas.extend(m)
    
    ## Remove duplicate
    docs_unique = []
    metas_unique = []
    duplist = []
    
    for idx1, p in enumerate(tqdm(docs)):
        sub_list = docs[idx1 + 1:]
        is_duplicate = False
        for q in sub_list:
            distance = textdistance.hamming.distance(p, q)
            distance = distance / max(len(p), len(q))
            if distance < SIMILARITY_THRESHOLD:
                duplist.append({'A':p, 'B':q, 'distance': distance})
                is_duplicate = True
                break
        if not is_duplicate:
            docs_unique.append(p)
            metas_unique.append(metas[idx1])
    
    print('Total Docs: ', len(docs))
    print('Duplicates: ', len(docs)-len(docs_unique))
    print('Unique Docs: ', len(docs_unique))
    
    pd.DataFrame({
        'fname':[x['fname'] for x in metas_unique],
        'id': [x['id'] for x in metas_unique],
        'context':[x['context'] for x in metas_unique],
        'question':[x['question'] for x in metas_unique],
        'answer':[x for x in metas_unique],
    }).to_csv('./docx_context.csv', index=False, encoding='utf-8')
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators = separators,
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = e5_len,
    )
    docs_unique = text_splitter.create_documents(docs_unique, metadatas=metas_unique)
    docs_unique = text_splitter.split_documents(docs_unique)
    
    # Filter chunk with duplicates
    final_docs = []
    for idx1, p in enumerate(tqdm(docs_unique)):
        sub_list = docs_unique[idx1 + 1:]
        is_duplicate = False
        for q in sub_list:
            distance = textdistance.hamming.distance(p.page_content, q.page_content)
            distance = distance / max(len(p.page_content), len(q.page_content))
            if distance < SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            final_docs.append(p)
            
    print('Duplicate Chunks: ', len(docs_unique)-len(final_docs))
    return final_docs



## Embedding Functions
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embedding(texts, model, tokenizer):
    with torch.no_grad():
        batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        for k, v in batch_dict.items():
            batch_dict[k] = batch_dict[k].to('cuda')
        
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings
    
def get_scores(query, key):
    query = F.normalize(query, p=2, dim=1)
    key = F.normalize(key, p=2, dim=1)
    scores = (query @ key.T) * 100
    return scores.tolist()

def compute_all_embedding(texts, model, tokenizer, bs=16):
    n = len(texts) // bs + 1
    res = []
    for i in tqdm(range(n)):
        batch = texts[i*bs:i*bs+bs]
        if len(batch) == 0: continue
        embed = get_embedding(batch, model, tokenizer)
        res.extend(embed)
        
    res = torch.stack(res, dim=0)
    return res



#@st.cache_resource()
def load_model(data_path, chunk_size, chunk_overlap, separators, device):
    if not os.path.exists('./.cache'):
        os.mkdir('./.cache')
    
    # Load embedding model
    tokenizer_emb = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model_emb = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(device)
    model_emb.eval()
    
    # Load documents
    print('>>>> Loading Documents')
    if os.path.exists(DOCS_CACHE_PATH):
        with open(DOCS_CACHE_PATH, 'rb') as handle:
            d = pickle.load(handle)
        docs = d['docs']
        embed = d['embed']
            
    else:
        docs = get_docs(data_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        
        data = []
        for d in docs:
            data.append('Question: ' + d.metadata['question'] + '\nAnswer: ' + d.page_content)
        
        print('>>>> Computing Embeddings')
        embed = compute_all_embedding(['passage: '+x for x in data], model_emb, tokenizer_emb)
        
        with open(DOCS_CACHE_PATH, 'wb') as handle:
            pickle.dump({'docs': docs, 'embed': embed}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print('>>>> Documents Loaded')
    print(docs[:5])
    return model_emb, tokenizer_emb, docs, embed




def embed_search(prompt, docs, embeddings, model_emb, tokenizer_emb, topn=16):
    qemb = get_embedding('query: '+prompt, model_emb, tokenizer_emb)
    prob = get_scores(qemb, embeddings)[0]
    idx = list(np.argpartition(prob, -topn)[-topn:])
    idx.sort(key=lambda x: prob[x], reverse=True)

    searched_context = np.array(docs)[idx].tolist()
    retrived_prob = [prob[x] for x in idx]

    result = [{'score': retrived_prob[idx], 'docs':searched_context[idx]} for idx, i in enumerate(searched_context)]
    return result




def embed_search_batch(queries:list, docs, embeddings, model_emb, tokenizer_emb, topn=16):
    qemb = get_embedding(['query: ' + x for x in queries], model_emb, tokenizer_emb)
    probs = get_scores(qemb, embeddings)
    
    results = []
    for prob in probs:
        idx = list(np.argpartition(prob, -topn)[-topn:])
        idx.sort(key=lambda x: prob[x], reverse=True)

        searched_context = np.array(docs)[idx].tolist()
        retrived_prob = [prob[x] for x in idx]

        result = [{'score': retrived_prob[idx], 'docs':searched_context[idx]} for idx, i in enumerate(searched_context)]
        results.extend(result)
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Filter duplicates
    final_results = []
    for r in results:
        found = False
        for fr in final_results:
            if r['docs'] == fr['docs']:
                found = True
        if found == False:
            final_results.append(r)
    return final_results