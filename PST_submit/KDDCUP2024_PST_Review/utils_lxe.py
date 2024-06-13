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

def remove_et_al_parentheses(text):
    pattern = re.compile(r'\([^)]*et al\.[^)]*\)')
    result = pattern.sub('', text)
    return result


def count_numbers_inside_brackets(text):
    pattern = re.compile(r'\[(.*?)\]')
    matches = pattern.findall(text)

    count = 0
    for match in matches:
        numbers = re.findall(r'\d+', match)
        count = max(len(numbers),count)

    return count

def find_bib_context(xml, dist=120):
    # path = "./knowledgeable_verbalizer.txt"
    # special_dict = {}
    # with open(path,"r") as file:
    #     for line in file:
    #         print()
    #         assert ()
            # special_dict[line.split()[0][:-1]] = line
            # print(special_dict[line[0]])
    # print(special_dict["method"])
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
    bracket_pattern = r'\[([^\]]+)\]'

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [ii for ii in range(len(xml)) if xml.startswith(item_str, ii)]
        match = re.search(pattern, item_str)
        item_st = f"[{match.group(1)}]"

        tmp = []
        # TODO:0525 find <head> </head>
        for head_pos in cur_bib_context_pos_start:
            # 查找每个引文位置最近的 head 标签内容
            text_up_to_position = xml[:head_pos]
            match = head_pattern.search(text_up_to_position)
            if match:
                tmp.append(match.group(1).strip())
            else:
                tmp.append("abstract")

        num_pos = 0
        if len(cur_bib_context_pos_start) >= 5:
            dist = 120
        else:
            dist = 200
        # dist_list = []
        # if len(cur_bib_context_pos_start) >= 4:
        #     number_less = int(len(cur_bib_context_pos_start)*0.33)
        #     dist_list = [100 for i in range(number_less)]
        #     dist_list.extend([100 for i in range(len(cur_bib_context_pos_start)-number_less*2)])
        #     dist_list.extend([150 for i in range(number_less)])
        #     # dist = 80
        # elif len(cur_bib_context_pos_start) == 3:
        #     dist_list = [150,200,150]
        #     # dist = 130
        # else:
        #     dist_list = [250,250]
        for i, pos in enumerate(cur_bib_context_pos_start):

            targer_context = xml[pos - dist: pos + dist]
            # 去换行
            targer_context = targer_context.replace("\n", " ").replace("\r", " ").strip().replace(item_str, '__REF__')
            # 去除"<--->"
            targer_context = re.sub(r'<[^>]*>', '', targer_context)
            # 去除特殊字符
            targer_context = re.sub(pattern_1, '', targer_context)
            # 去除多余参考文献编号干扰
            # targer_context = remove_numbers_inside_brackets(targer_context)
            # 提取出现位置

            #去除多余（et al. 2018a）
            # targer_context = remove_et_al_parentheses(targer_context)

            if fuzz.ratio("INTRODUCTION", tmp[num_pos]) < 30 and fuzz.ratio("Introduction", tmp[num_pos]) < 30:
                targer_context = "[pos is context] " + targer_context
                # targer_context = "[pos is in main body] " + targer_context
            # for word in targer_context.split():
            #     if word in special_dict["method"]:
            #         targer_context = "[cite role is method] " + targer_context
            #         break
            targer_context = targer_context.replace('__REF__', item_st)
            # bib_to_context[bib_id].append(targer_context)
            #TODO:0605
            # check number of citation [], if number over 3,then not consider to train
            flag = True
            flag_number = True
            # ——————————————————————————————————————————
            # le=0
            # ri=0
            # for i in range(0,len(targer_context)):
            #     if targer_context[i]=="[":
            #         le+=1
            #     elif targer_context[i]=="]":
            #         ri+=1
            #     # ori is >4
            #     if(le==ri and le>4):
            #         flag = False
            #         break
            # -------------------------------------------
            # if(flag):
                # bracket_pattern = r'\[([^\]]+)\]'
                # number_pattern = r'\d+'
                # bracket_contents = re.findall(bracket_pattern, targer_context)
                # all_numbers = []
                # for content in bracket_contents:
                #     # 从方括号内的内容中找到所有数字
                #     numbers = re.findall(number_pattern, content)
                #     all_numbers.extend(numbers)
                # # 统计数字的个数
                # # count = len(all_numbers)
                # # if count<=4:
                # #     bib_to_context[bib_id].append(targer_context)
                # # TODO:0607 ori is below
                # if(count_numbers_inside_brackets(targer_context)<=4):
                #     bib_to_context[bib_id].append(targer_context)
                # bib_to_context[bib_id].append(targer_context)
            bib_to_context[bib_id].append(targer_context)
            # num_pos += 1
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

# epoch = 2 4341   //  4 4607
# step = 3