# import torch
import logging
import warnings
warnings.filterwarnings("ignore")

import cfg
from sentence_transformers.cross_encoder import CrossEncoder
# from bi_encoder import get_bank_answers

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = cfg.device


# turn on logs
logging.basicConfig(level=logging.INFO)

# load model, tokenizer, embeddins, data
cross_encoder = CrossEncoder(cfg.name_cross_encoder)
logging.info('load cross-encoder - ok')

def re_rank(query_list, answers):
    # answers = get_bank_answers(query_list)
    query_list = [query.lower() for query in query_list]
    query = ' '.join(query_list)

    cross_inp = [[query, answer] for answer in answers]
    cross_scores = cross_encoder.predict(cross_inp)

    hits = dict(zip(answers, cross_scores))
    hits = sorted(hits.items(), key=lambda item: item[1], reverse=True)
    answer = hits[0][0]
    logging.info('re-rank answers - ok')
    return answer
