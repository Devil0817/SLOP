#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@project: cm-test
@author: minglu liu
@contact:  liuminglu@chinamobile.com
@file: task_utils.py
@time: 2023/9/1 14:47
@version: 1.0.0
"""
import os

from utils.misc import read_json


def find_unexec_tasks(ck_name, data_dir, result_dir, task_config):
    all_task_list = []
    group_list = os.listdir(data_dir)
    for group in group_list:
        test_list = os.listdir(os.path.join(data_dir, group))
        for test in test_list:
            if test not in task_config:
                continue
            task_list = os.listdir(os.path.join(data_dir, group, test))
            for task in task_list:
                if not task.endswith('.json'):
                    continue
                task_path = os.path.join(data_dir, group, test, task)
                sample_count = len(read_json(task_path))
                all_task_list.append({
                    'group': group,
                    'task': task,
                    'test': test,
                    'count': sample_count
                })
    unexec_task_list = []

    for task_item in all_task_list:
        group = task_item['group']
        task = task_item['task']
        test = task_item['test']
        count = task_item['count']
        task_path = os.path.join(result_dir, ck_name, group, test, task)
        if not os.path.exists(task_path):
            unexec_task_list.append(task_item)
        else:
            pass
            # sample_count = len(read_json(task_path))
            # if sample_count == count:
            #     continue
            # unexec_task_list.append(task_item)
    return unexec_task_list
