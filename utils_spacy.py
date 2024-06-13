from os.path import join
import json
import numpy as np
import pickle
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging

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

def clean_str(text,item_str):
    # pattern = r'target="(#[^"]+)"'
    # match = re.search(pattern, item_str)
    # item_st = f"[{match.group(1)}]"
    text = text.replace("\n", " ").replace("\r", " ").strip().replace(item_str, '__REF__')
    pattern_1 = r'[^a-zA-Z0-9.,!?\'" _()[\]]'
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(pattern_1, '', text)
    text = remove_numbers_inside_brackets(text)
    text = text.replace("__REF__",item_str)
    return text

def has_two_or_more_pairs(text):
    pairs = re.findall(r'\[[^\]]*\]', text)
    return len(pairs) >= 2

def find_bib_context(xml,splitter):
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

    xml = re.sub("<ref","__REF__",xml)
    xml = re.sub("</ref>","__REFF__",xml)
    bs = BeautifulSoup(xml, "xml")
    divs = bs.find_all("div")
    div_contents = []
    for div in divs:
        if "xmlns" not in div.attrs:
            continue
        div_contents.append(div.text.strip().replace("__REF__","<ref").replace("__REFF__","</ref>"))

    passage_sentences = []
    for item_div in div_contents:
        sentences = splitter.split_text(item_div)
        passage_sentences.append(sentences)

    pattern = r'target="(#[^"]+)"'

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        # match = re.search(pattern, item_str)
        # item_st = f"[{match.group(1)}]"
        for sentences in passage_sentences:
            # sentences = splitter.split_text(item_div)
            sentences = [clean_str(k,item_str) for k in sentences]
            for j,sentence in enumerate(sentences):
                # sentences_select = []
                if item_str in sentence:
                    # bib_to_context[bib_id].append(" ".join(sentences[max(0,j-1):min(j+2,len(sentences))]))
                    # if j-1 >=0:
                    #     sentences_select.append(sentences[j-1])

                    # sentences_select.append(sentence)

                    # if j+1 < len(sentences):
                    #     sentences_select.append(sentences[j+1])
                        
                    bib_to_context[bib_id].append(sentence)

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