# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:23:12 2021

@author: Jayanth
"""


from pytorch_transformers import BertTokenizer

import torch

def truncate_seq_pair(tokens_a,tokens_b,max_length:int):
    
    while True:
        total_length = len(tokens_a)+len(tokens_b)
        
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            

def sentence_pair_processing(data: list,tokenizer: BertTokenizer, max_sequence_length=128):
        
    max_bert_input_length = 0
    for sentence_pair in data:
        
        sentence_1_tokenized,sentence_2_tokenized = tokenizer.tokenize(sentence_pair['sentence_1']),tokenizer.tokenize(sentence_pair['sentence_2'])
        truncate_seq_pair(sentence_1_tokenized,sentence_2_tokenized,max_sequence_length-3)
        
        max_bert_input_length = max(max_bert_input_length, len(sentence_1_tokenized) + len(sentence_2_tokenized) + 3)
        sentence_pair['sentence_1_tokenized'] = sentence_1_tokenized
        sentence_pair['sentence_2_tokenized'] = sentence_2_tokenized
        
        dataset_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        dataset_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        dataset_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        dataset_scores = torch.empty((len(data), 1), dtype=torch.float)
        
    for idx, sentence_pair in enumerate(data):
        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sentence_pair['sentence_1_tokenized']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in sentence_pair['sentence_2_tokenized']:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_masks = [1] * len(input_ids)
        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)

        dataset_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        dataset_token_type_ids[idx] = torch.tensor(input_type_ids, dtype=torch.long)
        dataset_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)
        if 'similarity' not in sentence_pair or sentence_pair['similarity'] is None:
            dataset_scores[idx] = torch.tensor(float('nan'), dtype=torch.float)
        else:
            dataset_scores[idx] = torch.tensor(sentence_pair['similarity'], dtype=torch.float)

    return dataset_input_ids, dataset_token_type_ids, dataset_attention_masks, dataset_scores
        
        
        
        
        
        