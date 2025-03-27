#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@project: lc-data-preprocess
@author: minglu liu
@contact:  liuminglu@chinamobile.com
@file: corpus_utils.py
@time: 2023/7/7 14:44
@version: 1.0.0
"""
import os
from utils.misc import listdir


def fill_json_list(json_list, corpus_total_tokens, min_tokens=100_000_000_000, token_bytes=4):
    for json_object in json_list:
        dir_list = json_object['input_dir']

        bin_file_list = []
        for file_path in dir_list:
            bin_file_list.extend(listdir(file_path, suffix='.bin'))
        bin_file_list = set(bin_file_list)

        dataset_file_list = []
        for bin_file_path in bin_file_list:
            file_path = bin_file_path[0:-4]
            dataset_file_list.append(file_path)
        json_object['data_file_list'] = dataset_file_list

        if json_object['total_tokens'] is None and json_object['epoch'] is not None:
            total_tokens = sum([os.path.getsize(bin_file) for bin_file in bin_file_list]) // token_bytes
            json_object['total_tokens'] = int(total_tokens * json_object['epoch'])
            json_object['file_size'] = json_object['total_tokens'] * token_bytes / 1024 / 1024 / 1024

    for json_object in json_list:

        if json_object['total_tokens'] is not None:
            continue

        for json_object in json_list:
            if json_object['total_tokens'] is None:
                continue
            else:
                corpus_total_tokens -= json_object['total_tokens']

        if corpus_total_tokens < min_tokens:
            corpus_total_tokens = min_tokens

        json_object['total_tokens'] = corpus_total_tokens
        json_object['file_size'] = json_object['total_tokens'] * 4 / 1024 / 1024 / 1024
