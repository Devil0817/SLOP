#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@project: cm-test
@author: minglu liu
@contact:  liuminglu@chinamobile.com
@file: gpu_utils.py
@time: 2023/8/30 9:12
@version: 1.0.0
"""
import re
import subprocess
import sys


def get_gpu_name_memory():
    if sys.platform == 'win32':
        output = subprocess.Popen(
            "wmic path win32_VideoController get Name, AdapterRAM",
            stdout=subprocess.PIPE,
            shell=True
        ).communicate()[0].decode()
    else:
        output = subprocess.Popen(
            "nvidia-smi",
            stdout=subprocess.PIPE,
            shell=True
        ).communicate()[0].decode()
    lines = output.strip().split("------------")
    gpu_stat = {}

    for line in lines:
        if "NVIDIA A100-SXM" in line:
            mem = re.findall('[\d]+MiB / [\d]+MiB', line)[0]
            mem = float(mem.split('MiB')[0])
            gpu_stat[len(gpu_stat)] = mem
            # gpu_stat[len(gpu_stat)] = 0
    return gpu_stat


def report_memory(name=''):
    """Simple GPU memory report."""
    import torch

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max cached: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print(string, flush=True)
    return string