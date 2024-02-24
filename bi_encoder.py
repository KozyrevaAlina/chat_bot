import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
import numpy as np
import logging
import cfg
from utils.bi_encode import encode
import warnings
warnings.filterwarnings("ignore")

device = cfg.device

logging.info('device {device}')
# turn on logs
logging.basicConfig(level=logging.INFO)

# load model, tokenizer, embeddins, data
model = AutoModel.from_pretrained(cfg.name_model).to(device)
logging.info('load bi-encoder - ok')

tokenizer = AutoTokenizer.from_pretrained(cfg.name_tokenizer)   
logging.info('load tokenizer - ok')

loaded_response_embeddings = np.load(cfg.response_embeddings)
logging.info('load embeddings - ok')

loaded_responses = []
with open(cfg.responses, "r", encoding="utf-8") as file:
    loaded_responses = [line.strip() for line in file.readlines()]
logging.info('load bank of responses - ok')

# answer the query
def get_best_answer(query_list, 
                    responses = loaded_responses, 
                    response_embeddings = loaded_response_embeddings):
    
    '''
    Получаем один ответ с наивысшей оценкой

    Параметры:
    query_list - список реплик (вопрос с контекстом)
    responses - банк ответов
    response_embeddings - эмбеддинги ответов

    Возвращает:
    answer - один ответ
    '''
        
    query_list = [query.lower() for query in query_list]
    query = ' '.join(query_list)


    # Encode the query using the bi-encoder and find potentially relevant answer
    question_embedding = encode(query, tokenizer, model, device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        question_embedding = question_embedding.cuda()  
    hits = util.semantic_search(question_embedding, response_embeddings, top_k=32)
    hits = hits[0]  # Get the hits for the first query

    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    hit = hits[0]
    answer = "{}".format(responses[hit['corpus_id']].replace("\n", " "))
    if answer == 'nan':
        answer = 'I do not know'
    logging.info('answer is ready - ok')    
    return answer 

# answer the query
def get_bank_answers(query_list, 
                    responses = loaded_responses, 
                    response_embeddings = loaded_response_embeddings,
                    top_k = 32):
    '''
    Получаем список top_k ответов с наивысшей оценкой

    Параметры:
    query_list - список реплик (вопрос с контекстом)
    responses - банк ответов
    response_embeddings - эмбеддинги ответов
    top_k - сколько ответов возвращает фугкция

    Возвращает:
    answers - список из top_k ответов
    '''
    answers = []
    query_list = [query.lower() for query in query_list]
    query = ' '.join(query_list)

    # Encode the query using the bi-encoder and find potentially relevant answer
    question_embedding = encode(query, tokenizer, model, device)
    if device == 'cuda':
        torch.cuda.empty_cache()
        question_embedding = question_embedding.cuda()  
    hits = util.semantic_search(question_embedding, response_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    hit = hits[0]
    for hit in hits:
        answer = "{}".format(responses[hit['corpus_id']].replace("\n", " "))
        if answer == 'nan':
            answer = 'I do not know'
        answers.append(answer)
    logging.info('answer is ready - ok')    
    return answers 

