import json 
from os.path import abspath, dirname, join
PROJ_DIR = join(abspath(dirname(__file__)))

# with open("/data/zsp/KDD_PST_2341/paper-source-trace/out/kddcup/sciroberta_0.44564_add_pos/test_submission_addpos.json",'r') as f:
#     data_bert_1 = json.load(f)

with open(f"{PROJ_DIR}/result/gcn_fold3.json",'r') as f:
    data_gcn = json.load(f)

with open(f"{PROJ_DIR}/result/roberta_spacy_fold2.json",'r') as f:
    data_bert_2 = json.load(f)

with open(f"{PROJ_DIR}/result/test_ab_div_sciroberta.json",'r') as f:
    data_jss = json.load(f)

with open(f"{PROJ_DIR}/result/test_lxe_sciroberta.json",'r') as f:
    data_lxe = json.load(f)

with open(f"{PROJ_DIR}/data/PST/Btest_monogr_mask.json",'r') as f:
    masks = json.load(f)

lis = data_bert_2.keys()

result={}

for key in lis:
    # item_bert_1 = data_bert_1[key]
    item_data_gcn = data_gcn[key]
    item_bert_2 = data_bert_2[key]
    item_data_lxe = data_lxe[key]
    item_data_jss = data_jss[key]
    mask = masks[key]
    for idx,ma in enumerate(mask):
        if not ma:
            # item_bert_1[idx] = 0.0
            item_data_gcn[idx] = 0.0
            item_bert_2[idx] = 0.0
            item_data_lxe[idx] = 0.0
            item_data_jss[idx] = 0.0
    result_1 = []
    for idx,ma in enumerate(mask):
        result_1.append(0.2*item_data_gcn[idx]+0.4*item_bert_2[idx]+0.2*item_data_lxe[idx]+0.2*item_data_jss[idx])
    result[key]=result_1

with open(f"{PROJ_DIR}/result/test_submission.json",'w') as f:
    json.dump(result,f,indent=4)