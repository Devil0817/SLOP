#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@project: check-dul
@author: minglu liu
@contact:  liuminglu@chinamobile.com
@file: text_utils.py
@time: 2023/3/8 17:07
@version: 1.0.0
"""

j_start = ord('\u3040')
j_end = ord('\u30FF')


def is_japanese(uchar):
    if uchar >= j_start and uchar <= j_end:
        return True
    else:
        return False


c_start = ord('\u4e00')
c_end = ord('\u9fa5')


def is_chinese(uchar):
    # pattern_num_comma = r"[\u4E00-\u9FA5]"
    # return re.match(pattern_num_comma, char)
    if uchar >= c_start and uchar <= c_end:
        return True
    else:
        return False


def add_whitespace(line):
    line = line.strip()
    out_chars = []
    for c in line:
        try:
            uc = ord(c)
            if is_chinese(uc) or is_japanese(uc):
                out_chars.append(c)
                out_chars.append(' ')
            else:
                out_chars.append(c)
        except:
            pass
    return ''.join(out_chars)


common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                            '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]


def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total


num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
                        '十']
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']


def chinese_to_arabic(oriStr):
    lenStr = len(oriStr);
    aProStr = ''
    if lenStr == 0:
        return aProStr;

    hasNumStart = False;
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True;

            numberStr += oriStr[idx]
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False;
                    aProStr += numResult

            aProStr += oriStr[idx]
            pass

    if len(numberStr) > 0:
        resultNum = chinese2digits(numberStr)
        aProStr += str(resultNum)

    return aProStr


def pattern_transform(pattern_list, json_object):
    """

    :param pattern: ['ori_key1#map_key1', 'ori_key2#map_key2']
    :param json_object:
    :return:
    """
    text = []
    for pattern in pattern_list:
        ori_key = pattern.split('#')[0]
        map_key = pattern.split('#')[-1]
        if len(map_key) > 0 and not map_key.endswith(':'):
            map_key += ':'
        sub_text = json_object[ori_key]
        text.append(f'{map_key}{sub_text}')
    return ' '.join(text)


def zht2s(text):
    import zhconv
    text = zhconv.convert(text, 'zh-cn')
    return text


buffer_dict = {}


def get_chunk_idx(qid, chunk_num):
    from hashlib import md5
    if qid in buffer_dict:
        return buffer_dict[qid]

    md5_hashcode = md5(qid.encode(encoding='utf8', errors='ignore'))
    chunk_id = int(md5_hashcode.hexdigest(), 16) % chunk_num
    buffer_dict[qid] = chunk_id
    return chunk_id


def extract_text_from_html(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, features='lxml')
    elems = soup.find_all('p')
    content_lines = []
    for elem in elems:
        content_lines.append(elem.text)
    content_lines = [line for line in content_lines if len(line) > 0]
    content = '\n'.join(content_lines)
    return content
