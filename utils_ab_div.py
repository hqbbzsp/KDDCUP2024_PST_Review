from os.path import join
import json
import numpy as np
import pickle
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def remove_numbers_inside_brackets(text):
    pattern = re.compile(r'\[(.*?)\]')
    
    def replace_numbers(match):
        content = match.group(1)
        if re.fullmatch(r'\d+', content):
            return f'[{content}]'
        else:
            new_content = re.sub(r'\d+', '', content)
            new_content = re.sub(r',', '', new_content)
            return f'[{new_content}]'
    
    result = pattern.sub(replace_numbers, text)
    
    return result


def find_bib_context(xml, dist=100):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = dd(list)
    bibr_strs_to_bid_id = {}
    for item in bs.find_all(type='bibr'):
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]
        # item_str = "<ref type=\"bibr\" target=\"{}\">{}</ref>".format(item.attrs["target"], item.get_text())
        item_str = "<ref type=\"bibr\" target=\"{}\">".format(item.attrs["target"])
        bibr_strs_to_bid_id[item_str] = bib_id

    pattern = r'target="(#[^"]+)"'
    pattern_1 = r'[^a-zA-Z0-9.,!?\'" _()[\]]'
    head_pattern = re.compile(r'<head(?: n="[\d\.]+")?>(.+?)</head>')

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [ii for ii in range(len(xml)) if xml.startswith(item_str, ii)]
        match = re.search(pattern, item_str)
        item_st = f"[{match.group(1)}]"

        tmp = []
        #TODO:0525 find <head> </head>
        for head_pos in cur_bib_context_pos_start:
            # 查找每个引文位置最近的 head 标签内容
            text_up_to_position = xml[:head_pos]
            match = head_pattern.search(text_up_to_position)
            if match:
                tmp.append(match.group(1).strip())
            else:
                tmp.append("abstract")

        num_pos=0
        for i,pos in enumerate(cur_bib_context_pos_start):
            targer_context = xml[pos - dist: pos + dist]
            # 去换行
            targer_context = targer_context.replace("\n", " ").replace("\r", " ").strip().replace(item_str, '__REF__')
            # 去除"<--->"
            targer_context = re.sub(r'<[^>]*>', '', targer_context)
            # 去除特殊字符
            targer_context = re.sub(pattern_1, '', targer_context)
            # 去除多余参考文献编号干扰
            targer_context = remove_numbers_inside_brackets(targer_context)
            # 提取出现位置
            if fuzz.ratio("INTRODUCTION",tmp[num_pos]) <30 and fuzz.ratio("Introduction",tmp[num_pos]) <30:
                targer_context = "[pos is " + tmp[num_pos] + "] " +targer_context
            targer_context = targer_context.replace('__REF__', item_st)
            bib_to_context[bib_id].append(targer_context)
    return bib_to_context


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Log:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(file_path, 'w+')

    def log(self, s):
        self.f.write(str(datetime.now()) + "\t" + s + '\n')
        self.f.flush()