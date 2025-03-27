#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@project: cm-test
@author: minglu liu
@contact:  liuminglu@chinamobile.com
@file: worker_utils.py
@time: 2023/9/6 17:04
@version: 1.0.0
"""
from datetime import datetime


def is_state_expired(state_dict, key):
    if state_dict[key]['state'] != 'done' and diff_time_by_day(
            get_current_datetime(), state_dict[key]['update_time']) > 2:
        return True
    else:
        return False


import math


def diff_time_by_day(time_str1, time_str2):
    time1 = datetime.strptime(time_str1, '%Y-%m-%d %H:%M:%S')
    time2 = datetime.strptime(time_str2, '%Y-%m-%d %H:%M:%S')
    diff = time1 - time2
    return math.fabs(diff.days)


def get_current_datetime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
