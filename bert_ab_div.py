import os
import sys
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
# from transformers import SequenceClassification
from transformers import RobertaForSequenceClassification
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader #, SequentialSampler
from tqdm import trange
from sklearn.metrics import average_precision_score #classification_report, precision_recall_fscore_support, 
import logging

import utils_ab_div as utils
import settings
# import json
import random

# 定义随机种子固定的函数
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MAX_SEQ_LENGTH=512


def prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    # random.seed(2024)
    # random.shuffle(papers)
    n_train = int(n_papers * 2 / 3)
    # n_valid = n_papers - n_train

    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    pid_to_title = {}
    for paper in tqdm(papers):
        pid = paper["_id"]
        pid_to_title[pid] = paper["title"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    count=0
    all_pids = list(pids_train | pids_valid)
    random.shuffle(all_pids)
    for cur_pid in tqdm(all_pids):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
        count+=1
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        flag = False

        cur_pos_bib = set()

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)
        
        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        
        if not flag:
            continue
    
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
    
        bib_to_contexts = utils.find_bib_context(xml,200)

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)
        # is_valid=False

        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
        elif cur_pid in pids_valid:
            # is_valid=True
            cur_x = x_valid
            cur_y = y_valid
        else:
            continue
            # raise Exception("cur_pid not in train/valid/test")
        orig_title = pid_to_title[cur_pid]

        # pos_id = list(cur_pos_bib)[0]
        # pos_context=bib_to_contexts[pos_id]
        # pos_title = bid_to_title[pos_id]
        
        for bib in cur_pos_bib:
            cur_title = bid_to_title[bib]
            cur_context = ". ".join(bib_to_contexts[bib])
            # if cur_context:
            cur_x.append("<p> "+orig_title+" </p><p> "+cur_title+" </p><p> "+cur_context+" </p>")
            cur_y.append(1)
    
        for i,bib in enumerate(cur_neg_bib_sample):
            cur_title = bid_to_title[bib]
            cur_context = ". ".join(bib_to_contexts[bib])
            # if cur_context:
            cur_x.append("<p> "+orig_title+" </p><p> "+cur_title+" </p><p> "+cur_context+" </p>")
            cur_y.append(0)

                # if i % 3==0 and not is_valid:
                #     cur_context = " ".join(pos_context[:3])
                #     cur_context += " ".join(bib_to_contexts[bib][:3])
                #     cur_x.append("<p> "+orig_title+" </p><p> "+pos_title+" </p><p> "+cur_title+" </p><p> "+cur_context+" </p>")
                #     cur_y.append(0.5)


        if count % 100 ==1:
            print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))
            with open(join(data_dir, "bib_context_train.txt"), "w", encoding="utf-8") as f:
                for line in x_train:
                    f.write(line + "\n")
            
            with open(join(data_dir, "bib_context_valid.txt"), "w", encoding="utf-8") as f:
                for line in x_valid:
                    f.write(line + "\n")
            
            with open(join(data_dir, "bib_context_train_label.txt"), "w", encoding="utf-8") as f:
                for line in y_train:
                    f.write(str(line) + "\n")
            
            with open(join(data_dir, "bib_context_valid_label.txt"), "w", encoding="utf-8") as f:
                for line in y_valid:
                    f.write(str(line) + "\n")
    
    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))


    with open(join(data_dir, "bib_context_train.txt"), "w", encoding="utf-8") as f:
        for line in x_train:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_context_valid.txt"), "w", encoding="utf-8") as f:
        for line in x_valid:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_context_train_label.txt"), "w", encoding="utf-8") as f:
        for line in y_train:
            f.write(str(line) + "\n")
    
    with open(join(data_dir, "bib_context_valid_label.txt"), "w", encoding="utf-8") as f:
        for line in y_valid:
            f.write(str(line) + "\n")


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    exceed = 0
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            # input_ids = input_ids[:max_seq_length]
            half = len(input_ids)//2
            input_ids = input_ids[(half-256):(half+256)]
            exceed = exceed + 1

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return input_items,exceed


def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device, criterion):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    y_pre = []
    y_true = []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            r = model(input_ids, attention_mask=input_mask,
                      token_type_ids=segment_ids, labels=label_ids)
            # tmp_eval_loss = r[0]
            logits = r[1]
            # print("logits", logits)
            tmp_eval_loss = criterion(logits, label_ids)

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()

        predicted_labels += list(outputs)
        correct_labels += list(label_ids)

        logits = torch.softmax(logits, dim=-1)
        y_pre.extend(logits[:, 1].cpu().detach().numpy().tolist())
        y_true.extend(label_ids.tolist())
        # print(y_pre)
        # print(y_true)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    try:
        eval_ap = average_precision_score(y_true, y_pre)
        print("Eval ap:", eval_ap)
    except ValueError :
        with open("a.txt","w") as f:
            print(y_true,file=f)
            print(y_pre,file=f)
        assert()
    eval_loss = eval_loss / nb_eval_steps
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)

    model.train()

    return eval_loss, correct_labels, predicted_labels,eval_ap


def train(year=2023, model_name="sciroberta"):
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    # data_year_dir = '/data/zsp/KDD_PST_2341/paper-source-trace/data/PST'
    print("data_year_dir", data_year_dir)

    with open(join(data_year_dir, "bib_context_train.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_texts.append(line.strip())
    with open(join(data_year_dir, "bib_context_valid.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_texts.append(line.strip())

    with open(join(data_year_dir, "bib_context_train_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_labels.append(float(line.strip()))
    with open(join(data_year_dir, "bib_context_valid_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_labels.append(float(line.strip()))


    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # np.bincount(train_labels): 这个函数用于计算数组中每个值的出现次数。
    # 假设train_labels中的类别标签从0开始且连续（如0, 1, 2, ...），
    # np.bincount将返回一个数组，其中每个元素的索引代表类别标签，其值为该类别的样本数量。
    # 权重与频率成反比
    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    print(np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    # [0.5500, 5.5000]

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "sciroberta":
        #allenai/cs_roberta_base
        BERT_MODEL = "allenai/cs_roberta_base"
        # BERT_MODEL = "/data/zsp/KDD_PST_2341/models/sci-roberta"
    else:
        raise NotImplementedError
    # 分词器（tokenizer）
    # 加载的分词器主要用于文本预处理。它将原始文本数据转换成模型可以理解的格式
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # 用于基于BERT架构的序列分类任务。序列分类通常是指对整个输入序列（例如一句话或一段文本）分配一个类别标签的任务。
    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features,exceed = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    print("train exceed number is : ",exceed)
    dev_features,exceed = convert_examples_to_inputs(dev_texts, dev_labels, MAX_SEQ_LENGTH, tokenizer)
    print("valid exceed number is : ", exceed)
    # 原16
    BATCH_SIZE = 32
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    # 指定了梯度累积的步数。设置为1意味着每个批次后都会更新模型（没有累积）。
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 10
    LEARNING_RATE = 5e-6
    WARMUP_PROPORTION = 0.1
    # 用于梯度裁剪，防止在训练过程中出现梯度爆炸问题。这里设置为5，表示如果梯度的范数超过5，将会被缩放回5。
    MAX_GRAD_NORM = 5
    EVAL_STEP = 100

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)

    OUTPUT_DIR = join(settings.OUT_DIR, "kddcup", model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_FILE_NAME = "pytorch_model_ab_div_1.bin"
    PATIENCE = 5

    eval_ap_history = []
    loss_history = []
    ap_history = []
    no_improvement = 0
    model.train()
    all_steps=0
    for epoch in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        tr_loss = 0
        # nb_tr_examples, nb_tr_steps = 0, 0
        y_true = []
        y_pre = []
        flag=False
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training iteration")
        for step, batch in enumerate(progress_bar):
            all_steps+=1
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            # loss = outputs[0]
            logits = outputs[1]

            loss = criterion(logits, label_ids)
            # TODO:0516 原是logits全softmax,改成sigmoid看下效果，因为原先的loss很奇怪
            # TODO:baseline代码如下
            logits = torch.softmax(logits, dim=-1)

            # logits[:,1] = torch.sigmoid(logits[:,1])
            # print(logits)
            y_pre.extend(logits[:, 1].cpu().detach().numpy().tolist())
            y_true.extend(label_ids.cpu().detach().numpy().tolist())
            # print(len(y_pre))
            # print(len(y_true))

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # MAX_GRAD_NORM = 5
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if all_steps% EVAL_STEP == 0 and all_steps >= 200:
                train_ap = average_precision_score(y_true, y_pre)
                dev_loss, _, _,eval_ap = evaluate(model, dev_dataloader, device, criterion)
                print("---------")
                print("Eval Loss history:", loss_history)
                print("Train AP history:", ap_history)
                print("Eval AP history:", eval_ap_history)
                print("Eval loss:", dev_loss)
                print("Train ap:", train_ap)
                print("Eval ap:", eval_ap)

                if len(eval_ap_history) == 0 or eval_ap > max(eval_ap_history):
                    no_improvement = 0
                    # 并行来利用多个GPU,需要访问.module来获取原始的未封装模型。
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    with open(os.path.join(OUTPUT_DIR,"score.txt"),"w") as f:
                        f.write("loss_history\n")
                        f.write(str(loss_history)+"\n")
                        f.write("train ap_history\n")
                        f.write(str(ap_history)+"\n")
                        f.write("eval ap_history\n")
                        f.write(str(eval_ap_history) + "\n")
                else:
                    no_improvement += 1

                if no_improvement >= PATIENCE:
                    with open(os.path.join(OUTPUT_DIR,"score.txt"),"w") as f:
                        f.write("loss_history\n")
                        f.write(str(loss_history)+"\n")
                        f.write("train ap_history\n")
                        f.write(str(ap_history)+"\n")
                        f.write("eval ap_history\n")
                        f.write(str(eval_ap_history) + "\n")
                    print("No improvement on development set. Finish training.")
                    flag=True
                    break

                loss_history.append(dev_loss)
                ap_history.append(train_ap)
                eval_ap_history.append(eval_ap)
            progress_bar.set_postfix(loss=tr_loss/(step+1))
        if flag:
            break



def gen_kddcup_valid_submission_bert(model_name="scibert"):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "sciroberta":
        #allenai/cs_roberta_base
        BERT_MODEL = "allenai/cs_roberta_base"
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    label_mask = utils.load_json(data_dir, "Btest_monogr_mask.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model_ab_div_1.bin")))

    model.to(device)
    model.eval()
    # TODO:0510 原是16，但是predict 无所谓
    BATCH_SIZE = 16
    # metrics = []
    # f_idx = 0

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    all_exceed_num = 0
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        orig_title = paper["title"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        # 找目标paper有多少ref_paper
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            bid_to_title[bid] = ""
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_to_contexts= utils.find_bib_context(xml,200)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue

        indices = [i for i, item in enumerate(bib_sorted) if not bib_to_contexts[item]]

        contexts_sorted = ["<p> "+orig_title+" </p><p> "+bid_to_title[bib]+" </p><p> "+". ".join(bib_to_contexts[bib])+" </p>" for bib in bib_sorted]

        test_features,exceed = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        all_exceed_num = all_exceed_num + exceed

        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                          token_type_ids=segment_ids, labels=label_ids)
                # tmp_eval_loss = r[0]
                logits = r[1]

            logits = torch.softmax(logits, dim=-1)
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)

        for ind in indices:
            predicted_scores[ind]=0
        
        mask = label_mask[cur_pid]
        assert len(mask) == len(predicted_scores)
        result_temp=[]
        for mas,score in zip(mask,predicted_scores):
            if mas:
                result_temp.append(score)
            else:
                result_temp.append(0.0)
        
        for ii in range(len(result_temp)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            # y_score[bib_idx] = float(utils_spacy.sigmoid(predicted_scores[ii]))
            y_score[bib_idx] = float(result_temp[ii])
        
        sub_dict[cur_pid] = y_score
        # input()
    
    utils.dump_json(sub_dict, settings.RESULT_DIR, "test_ab_div_sciroberta.json")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Please specify 'train' or 'test' as a command-line argument.")
    
    get_random_seed(2024)
    train_or_test = sys.argv[1]
    if train_or_test == "train":
        prepare_bert_input()
        train(model_name="sciroberta")

    elif train_or_test == "test":
        gen_kddcup_valid_submission_bert(model_name="sciroberta")
    else:
        raise ValueError("Invalid argument. Please specify 'train' or 'test'.")
