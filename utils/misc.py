# -*- coding:utf-8 _*-
"""
@author:mlliu
@file: misc.py
@time: 2018/04/03
"""
import json
import os


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    print(*message, flush=True)


import numpy as np


def read_npy(file_path):
    obj = np.load(file_path, allow_pickle=True)
    dict_obj = obj.item()
    return dict_obj


def save_npy(file_path, obj):
    check_and_mkdirs(file_path)
    np.save(file_path, obj)


import pickle


def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(file_path, obj):
    check_and_mkdirs(file_path)
    pickle_protocol = 4
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle_protocol)


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        text = f.read()
    return text


def read_txt_lines(file_path):
    in_f = open(file_path, "r", encoding="utf8", errors='ignore')
    content = []
    while 1:
        buffer = in_f.read(8 * 1024 * 1024)
        if not buffer:
            break
        content.append(buffer)
    content = ''.join(content)
    lines = content.split("\n")
    return lines


def read_big_json_list(file_path):
    import ijson
    with open(file_path, "r", encoding='utf8', errors='ignore') as f:
        data = ijson.items(f, 'item')
        while True:
            try:
                yield data.__next__()
            except StopIteration as e:
                break


def read_json(file_path):
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        json_object = json.load(f)
    return json_object


def save_json(file_path, json_object):
    check_and_mkdirs(file_path)
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(json_object, f, ensure_ascii=False, indent=2)


def save_txt(file_path, lines):
    check_and_mkdirs(file_path)
    if isinstance(lines, list):
        lines = "\n".join(lines)
    elif isinstance(lines, set):
        lines = "\n".join(lines)
    else:
        lines = str(lines)
    with open(file_path, 'w', encoding='utf8') as f:
        f.writelines(lines)


def listdir(path, suffix=None, prefix=None):
    if not os.path.exists(path) or not os.path.isdir(path):
        return []
    allfile = []
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            allfile.extend(listdir(filepath, suffix, prefix))
        else:
            suffix_fit = True
            prefix_fit = True
            filename = os.path.split(filepath)[-1]
            if suffix is not None:
                if not filename.endswith(suffix):
                    suffix_fit = False
            if prefix is not None:
                if not filename.startswith(prefix):
                    prefix_fit = False
            if suffix_fit and prefix_fit:
                allfile.append(filepath)
    return allfile


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def read_json_list(file_path, verbose=False):
    if verbose:
        from tqdm import tqdm

        in_f = open(file_path, "r", encoding="utf8", errors='ignore')
        count = 0
        while 1:
            buffer = in_f.read(8 * 1024 * 1024)
            if not buffer:
                break
            count += buffer.count('\n')
        in_f.close()

    in_f = open(file_path, "r", encoding="utf8", errors='ignore')
    json_list = []
    if verbose:
        iter_bar = tqdm(in_f, total=count)
    else:
        iter_bar = in_f

    for line in iter_bar:
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        try:
            line = json.loads(line)
        except:
            line = eval(line)
        json_list.append(line)

    return json_list


def save_json_list(file_path, json_list, verbose=False):
    check_and_mkdirs(file_path)
    with open(file_path, 'w', encoding='utf8') as f:
        if verbose:
            iter_list = process_bar(json_list)
        else:
            iter_list = json_list
        for json_object in iter_list:
            json_str = json.dumps(json_object, ensure_ascii=False)
            json_str = json_str + "\n"
            f.write(json_str)


def save_json_simplify(file_path, json_list):
    check_and_mkdirs(file_path)
    with open(file_path, 'w', encoding='utf8') as f:
        result_list = []
        for json_object in json_list:
            if 'head' in json_object and 'WARC-Target-URI' in json_object['head']:
                result_list.append(json_object['head']['WARC-Target-URI'])
        json_str = json.dumps(result_list, ensure_ascii=False)
        f.write(json_str)


def check_and_mkdirs(file_path):
    dir_path, file_name = os.path.split(file_path)

    if '.' not in file_name:
        dir_path = file_path

    if not os.path.exists(dir_path) and len(dir_path) != 0:
        os.makedirs(dir_path)


def download_file(url, file_path):
    import requests
    import os
    for _ in range(3):
        try:
            res = requests.get(url)
            _, file_name = os.path.split(file_path)
            check_and_mkdirs(file_path)
            with open(file_path, 'wb') as f:
                f.write(res.content)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
        break


def process_bar(elem_list: list):
    from tqdm import tqdm
    return tqdm(elem_list, total=len(elem_list))


def read_zst(file_path):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    result = []
    with open(file_path, 'rb') as ifh:
        reader = dctx.stream_reader(ifh, read_size=65536)
        data_chunk = reader.read(1024 * 1024)
        residual_line = ''
        while data_chunk:
            text_chunk = data_chunk.decode(encoding='utf8', errors='ignore')
            lines = (residual_line + text_chunk).split('\n')
            residual_line = lines[-1]

            for line in lines[0:-1]:

                json_object = json.loads(line)
                result.append(json_object)
                cur_doc: str = json_object['text']
                if len(cur_doc) == 0:
                    continue
            data_chunk = reader.read(1024 * 1024)
    return result


def read_zst_iterator(file_path):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with open(file_path, 'rb') as ifh:
        reader = dctx.stream_reader(ifh, read_size=65536)
        data_chunk = reader.read(1024 * 1024)
        residual_line = ''
        while data_chunk:
            text_chunk = data_chunk.decode(encoding='utf8', errors='ignore')
            lines = (residual_line + text_chunk).split('\n')
            residual_line = lines[-1]

            for line in lines[0:-1]:

                json_object = json.loads(line)
                yield json_object
                cur_doc: str = json_object['text']
                if len(cur_doc) == 0:
                    continue
            data_chunk = reader.read(1024 * 1024)
    return


def xml_reader(file_path):
    import xlrd
    workbook = xlrd.open_workbook(file_path, on_demand=True)  # 打开文件
    sheet_name = workbook.sheet_names()  # 所有sheet的名字
    sheets = workbook.sheets()  # 返回可迭代的sheets对象
    result_dict = {}
    for i, sheet in enumerate(sheets):
        name = sheet.name
        result_lines = []
        result_dict[name] = result_lines
        nrows = workbook.sheet_by_index(i).nrows
        ncols = workbook.sheet_by_index(i).ncols
        for p in range(nrows):
            line = {}
            for q in range(ncols):
                try:
                    str_value = workbook.sheet_by_index(i).cell_value(p, q)
                    line[q] = str_value
                except:
                    pass
            result_lines.append(line)
    return result_dict


def read_json_list_iterator(file_path):
    in_f = open(file_path, "r", encoding="utf8", errors='ignore')

    for line in in_f:
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        try:
            line = json.loads(line)
        except:
            try:
                line = eval(line)
            except:
                import traceback
                traceback.print_exc()
                continue

        yield line


def remove_duplicate(item_list):
    item_set = set()
    result_item_list = []
    for item in item_list:
        item_str = str(item)
        if item_str not in item_set:
            item_set.add(item_str)
            result_item_list.append(item)
    return result_item_list
