import boto3

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pickle
import json
import os

# Initialize a session using Amazon Bedrock
session = boto3.Session(
    aws_access_key_id= '',
    aws_secret_access_key='',
    region_name=''
)
bedrock_client = session.client('bedrock-runtime')

#Decide on the model to use
model_id = 'meta.llama2-13b-chat-v1'

#Create the sentence transformer model that you generate the text embeddings
embed_gen_model = SentenceTransformer('bert-base-nli-mean-tokens')

#docs are text-snippets/paragraphs from all the available text files
with open("./artifacts/docs.pkl", "rb") as f:
    docs = pickle.load(f)

#for each text-snippet in docs, doc_filenos consists of the file name that it belongs to
with open("./artifacts/doc_filenos.pkl", "rb") as f:
    doc_filenos = pickle.load(f)

#sentence-embeddings consist of the embeddings generated from sentence-transformer
with open("./artifacts/sentence_embeddings.pkl", "rb") as f:
    sentence_embeddings = pickle.load(f)

#extract text-snippets/paras from each document with the filename
def create_paras():
    files = os.listdir("./data/")
    docs = []
    doc_filenos = []
    for file in files:
        with open("./data/"+file, "r") as f:
            lines = f.readlines()

        res = []
        for line in lines:
            if len(line.split(" "))>10:
                res.append(line)

        docs += res
        doc_filenos += [file]*len(res)

    docs = dict(enumerate(docs))
    doc_filenos = dict(enumerate(doc_filenos))

    return docs, doc_filenos

#Find top k snippets that are relevant to the query using cosine similarity
def get_topk_similar_docs(query, sentence_embeddings, k=20):
    query_embedding = embed_gen_model.encode([query])
    scores = cosine_similarity(sentence_embeddings, query_embedding).flatten()
    topk = np.argsort(scores)[-k:]

    return topk

#compile context from all relevant text snippets
def create_context(docs_index):
    relevant_files = [doc_filenos[i] for i in docs_index]
    context = ""
    for file in relevant_files:
        with open("./data/"+file, "r") as f:
            content = f.read()
            context += (";" + content)

    context = context.replace('"', "'")
    context = context.replace('\"', "'")
    context = context.replace("\n", ";")
    context = context[:2000] 
    context.strip()

    return context

#use llama 2 for answering the query using the context extracted from the database
def ask_question(model_id, question, context):
    prompt = 'Answer the question: \'{}\' , \
    with the following context: \'{}\' \
    Only return the helpful answer below and nothing else.\
    Helpful answer:'.format(question, context)
    
    # Call the Bedrock service with the model ID and the payload
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType = "application/json",
        accept= "application/json",
        body='{"prompt":"'+prompt+'","max_gen_len":512,"temperature":0.5,"top_p":0.9}')
    
    return response

#perform Retrieval Augmented Generation to answer the given query using the knowledge base
def get_answer(query):

    try:
        topk = get_topk_similar_docs(query, sentence_embeddings, k=20)
        context = create_context(topk)

        response = ask_question(model_id, query, context)
        response = response["body"].read().decode()

        _json = json.loads(response)
        answer = _json["generation"]

        print(answer)
        return answer
    
    except Exception as e:
        print(e)
        return "Error processing the request!"