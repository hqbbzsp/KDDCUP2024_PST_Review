import copy
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
from transformers import RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from sklearn.metrics import average_precision_score
import logging
import random
import utils_lxe as utils
import settings
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 512
random.seed(2024)
os.environ['PYTHONHASHSEED'] = str(2024)
np.random.seed(2024)


def prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    # 788
    # print(n_papers)
    papers = sorted(papers, key=lambda x: x["_id"])

    n_train = int(n_papers * 2 / 3)
    # n_valid = n_papers - n_train
    # 2/3train  1/3val
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

    all_null_pos = 0
    all_null_neg = 0

    for cur_pid in tqdm(pids_train | pids_valid):
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        # 获取待查找 论文 名字
        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue
        # xml论文，再次获取相关信息
        references = bs.find_all("biblStruct")
        # 获取待寻找PST paper的xml所有索引论文
        bid_to_title = {}
        bid_to_publish_year = {}
        bid_to_quote_times = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]

            # 查看引用次数
            if (bid not in bid_to_quote_times):
                bid_to_quote_times[bid] = 1
            else:
                bid_to_quote_times[bid] += 1
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue

            bid_to_title[bid] = ref.analytic.title.text.lower()
            # B0 ,B1
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx
        flag = False

        cur_pos_bib = set()
        cur_pos_bib_year = []

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            # 可能不止 1篇 source_titles
            for label_title in source_titles:
                # rf_paper 对比 source_paper 相似度,单纯从文章名字，剔除一部分先
                # 打印出两个标题之间的相似度分数
                # 2024-05-05 排除引用不匹配的情况,即错误的源头论文？
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    # singan: learning a generative model from a single natural image
                    # singan: learning a generative model from a single natural image
                    # print(cur_ref_title," ",label_title)
                    flag = True
                    if bid not in cur_pos_bib:
                        cur_pos_bib.add(bid)
                        # cur_pos_bib_year.append(bid_to_publish_year[bid])

        # cur_pos_bib是 于sorce_paper名字相似的rf论文，设置为pos
        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
        # 从XML数据中提取带有引用标记的文本上下文
        bib_to_contexts= utils.find_bib_context(xml)
        n_pos = len(cur_pos_bib)
        # TODO:0506 因为重复采取，所以会有同样的片段出现，原始负样本数量如下
        n_neg = n_pos * 10

        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        if cur_pid in pids_train:
            cur_x = x_train
            cur_y = y_train
        elif cur_pid in pids_valid:
            cur_x = x_valid
            cur_y = y_valid
        else:
            continue
            # raise Exception("cur_pid not in train/valid/test")

        orig_title = pid_to_title[cur_pid]
        num = 0
        pos_null_count = 0
        neg_null_count = 0
        for bib in cur_pos_bib:

            # ---------------- ori
            cur_title = bid_to_title[bib]
            cur_context = " ".join(bib_to_contexts[bib])
            if(cur_context):
                cur_x.append(
                    "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                cur_y.append(1)
                num += 1

        num = 0
        for bib in cur_neg_bib_sample:

            # TODO:original is below
            # --------------------
            cur_title = bid_to_title[bib]
            cur_context = " ".join(bib_to_contexts[bib])

            if(cur_context):
                cur_x.append(
                    "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                cur_y.append(0)
                num += 1

    with open("./data_len.txt", "w") as f:
        print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid), file=f)
        print("all_null_pos is ",all_null_pos," all_null_neg is ",all_null_neg,file=f)
        # all_null_pos += 1
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


all_input_len = []
# train_features = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH=512, tokenizer, verbose=0)
def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    exceed = 0
    less_exceed = 0
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        # count null cnt, all cnt
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        all_input_len.append(len(input_ids))
        if len(input_ids) > max_seq_length:
            # ori
            # [0.5767 0.2434 0.3810] --> [0.5651 0.2882 0.3820]
            input_ids = input_ids[:max_seq_length]

            # middle
            # [0.5737 0.2680 0.4002]
            # half = len(input_ids) // 2
            # input_ids = input_ids[(half - 256):(half + 256)]
            #TODO:0603 try
            # input_ids = input_ids[:3] + input_ids[(half - 252):(half + 252)]
            exceed = exceed + 1

            # TODO: 0526 top and bowttom
            # 0.5491 0.2453 0.33339
            # len_input_ids = len(input_ids)
            # input_ids = input_ids[:250] + input_ids[len_input_ids-250:len_input_ids]
            # exceed = exceed + 1

        #delete too short item
        # elif len(input_ids) <30:
        #     less_exceed += 1
        #     continue
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

    return input_items, exceed ,less_exceed


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
        # TODO:0521 original is below
        logits = torch.softmax(logits, dim=-1)
        # logits[:, 1] = torch.sigmoid(logits[:, 1])
        y_pre.extend(logits[:, 1].cpu().detach().numpy().tolist())
        y_true.extend(label_ids.tolist())
        # print(y_pre)
        # print(y_true)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    try:
        eval_ap = average_precision_score(y_true, y_pre)
        print("Eval ap:", eval_ap)
    except ValueError as e:
        with open("a.txt", "w") as f:
            print(y_true, file=f)
            print(y_pre, file=f)
        assert ()
    eval_loss = eval_loss / nb_eval_steps
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)

    return eval_loss, correct_labels, predicted_labels, eval_ap

def train(year=2023, model_name="scibert"):
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)

    with open(join(data_year_dir, "bib_context_train.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_texts.append(line.strip())
    with open(join(data_year_dir, "bib_context_valid.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_texts.append(line.strip())

    with open(join(data_year_dir, "bib_context_train_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_labels.append(int(line.strip()))
    with open(join(data_year_dir, "bib_context_valid_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_labels.append(int(line.strip()))

    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # np.bincount(train_labels): 这个函数用于计算数组中每个值的出现次数。
    # 假设train_labels中的类别标签从0开始且连续（如0, 1, 2, ...），
    # np.bincount将返回一个数组，其中每个元素的索引代表类别标签，其值为该类别的样本数量。
    # 权重与频率成反比
    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    # [0.5500, 5.5000]

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
    # batch_size = 16  0.37863
    elif model_name == "sciroberta":
        BERT_MODEL = "allenai/cs_roberta_base"
        # BERT_MODEL = "/data/zsp/KDD_PST_2341/models/sci-roberta"
    # batch_size = 4  0.11
    elif model_name == "structbert":
        BERT_MODEL = "structbert"

    else:
        raise NotImplementedError
    # 分词器（tokenizer）
    # 加载的分词器主要用于文本预处理。它将原始文本数据转换成模型可以理解的格式
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # 用于基于BERT架构的序列分类任务。序列分类通常是指对整个输入序列（例如一句话或一段文本）分配一个类别标签的任务。
    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    # model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features, exceed ,less_exceed = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    print("train exceed number is : ", exceed)
    print("train less_exceed number is : ", less_exceed)
    dev_features, exceed ,less_exceed= convert_examples_to_inputs(dev_texts, dev_labels, MAX_SEQ_LENGTH, tokenizer)
    print("valid exceed number is : ", exceed)
    print("train less_exceed number is : ", less_exceed)
    # 原16
    # 由于 structbert太大，所以只能设成8，之前所有结果都是32
    BATCH_SIZE = 24
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    # 指定了梯度累积的步数。设置为1意味着每个批次后都会更新模型（没有累积）。
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 10
    LEARNING_RATE = 5e-6

    WARMUP_PROPORTION = 0.1
    # 用于梯度裁剪，防止在训练过程中出现梯度爆炸问题。这里设置为5，表示如果梯度的范数超过5，将会被缩放回5。
    MAX_GRAD_NORM = 5

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

    PATIENCE = 5
    EVAL_STEP = 200

    eval_ap_history = []
    loss_history = []
    ap_history = []

    step_eval_ap_history = []
    step_loss_history = []
    step_ap_history = []

    no_improvement = 0
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        y_true = []
        y_pre = []
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
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
            if (step+1) % EVAL_STEP == 0:
                train_ap = average_precision_score(y_true, y_pre)
                dev_loss, _, _, eval_ap = evaluate(model, dev_dataloader, device, criterion)
                print("----step-----")
                print("Loss step history:", step_loss_history)
                print("Train step AP history:", step_ap_history)
                print("eval step AP history:", step_eval_ap_history)
                print("Dev step loss:", dev_loss)
                print("Dev step ap:", train_ap)
                print("Eval step ap:", eval_ap)
                step_eval_ap_history.append(eval_ap)
                step_loss_history.append(dev_loss)
                step_ap_history.append(train_ap)

        train_ap = average_precision_score(y_true, y_pre)
        dev_loss, _, _, eval_ap = evaluate(model, dev_dataloader, device, criterion)
        print("----epoch-----")
        print("Loss history:", loss_history)
        print("Train AP history:", ap_history)
        print("eval AP history:", eval_ap_history)
        print("Dev loss:", dev_loss)
        print("Dev ap:", train_ap)
        print("Eval ap:", eval_ap)

        # model_to_save = model.module if hasattr(model, 'module') else model
        if dev_loss < min(loss_history) or len(loss_history)==0:
            model_to_save = model.module if hasattr(model, 'module') else model
            MODEL_FILE_NAME = "pytorch_model_lxe.bin"
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)

        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= PATIENCE:
            with open(f"./score_{model_name}.txt", "w") as f:
                f.write("loss_history\n")
                f.write(str(loss_history) + "\n")
                f.write("train ap_history\n")
                f.write(str(ap_history) + "\n")
                f.write("eval ap_history\n")
                f.write(str(eval_ap_history) + "\n")
                f.write("\n")
                f.write("---step---\n")
                f.write("step loss_history\n")
                f.write(str(step_loss_history) + "\n")
                f.write("train step ap_history\n")
                f.write(str(step_ap_history) + "\n")
                f.write("eval step ap_history\n")
                f.write(str(step_eval_ap_history) + "\n")

            print("No improvement on development set. Finish training.")
            break

        loss_history.append(dev_loss)
        ap_history.append(train_ap)
        eval_ap_history.append(eval_ap)

def gen_kddcup_valid_submission_bert(model_name="scibert"):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
    elif model_name == "sci-roberta":
        BERT_MODEL = "sci-roberta"
    elif model_name == "structbert":
        BERT_MODEL = "structbert"
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model.bin")))

    model.to(device)
    model.eval()
    # TODO:0510 原是16，但是predict 无所谓
    BATCH_SIZE = 24
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

        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue
        nu_special = 0
        special = []
        contexts_sorted = []
        # for bib in bib_sorted:
        #     special.append(len(bib_to_contexts[bib]))
        #     for j in bib_to_contexts[bib]:
        #         contexts_sorted.append(j)
        # TODO:0524 ori is below,change to [cal many line]max
        # contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        # cur_x.append(

        contexts_sorted = [
            "<title> " + orig_title + " </title><title> " + bid_to_title[bib] + " </title><context> " + " ".join(bib_to_contexts[bib]) + " </context>"
            for bib in bib_sorted]
        # contexts_sorted = ["[ori title: " + orig_title + "]" + " [ref title: " + bid_to_title[bib] + "] " + "cita context: " + " ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        test_features, exceed,less_exceed = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        all_exceed_num = all_exceed_num + exceed

        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                          token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = r[0]
                logits = r[1]

            logits = torch.softmax(logits, dim=-1)
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)

        # TODO:0524 ori is below,change to [cal many line]max
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            #TODO: original is below
            # y_score[bib_idx] = float(utils.sigmoid(predicted_scores[ii]))
            y_score[bib_idx] = float(predicted_scores[ii])
        sub_dict[cur_pid] = y_score

    # with open("./a.txt","w") as f:
    print("all test exceed number is : ", all_exceed_num)
    utils.dump_json(sub_dict, join(settings.OUT_DIR, "kddcup", model_name), f"valid_submission_{model_name}.json")

def draw_input_id():
    # 绘制直方图
    plt.hist(all_input_len, bins=5, edgecolor='black')
    # 添加标签和标题
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # 显示图表
    plt.show()


def Kfold_prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    # 788
    # print(n_papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_train = int(n_papers * 2 / 3)
    remain = len(papers) - n_train
    # n_valid = n_papers - n_train
    # 2/3train  1/3val
    first_ = papers[:remain]
    second_ = papers[remain:remain*2]
    third_ = papers[remain*2:]
    count = -1
    for i in range(0,3):
        count += 1
        # papers_train
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        if count==0:
            papers_train = first_+second_
            papers_valid = third_
        elif count==1:
            papers_train = first_ + third_
            papers_valid = second_
        elif count==2:
            papers_train = second_ + third_
            papers_valid = first_

        # print(len(papers_train))
        # print(len(papers_valid))
        # assert()
        # continue





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
        # all = 0
        # for cur_pid in tqdm(pids_train):
        #     all+= 1
        #     if(all) > 10:
        #         break

        all_null_pos = 0
        all_null_neg = 0

        for cur_pid in tqdm(pids_train | pids_valid):
            # cur_pid = file.split(".")[0]
            # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
            f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
            xml = f.read()
            bs = BeautifulSoup(xml, "xml")
            # 获取待查找 论文 名字
            source_titles = pid_to_source_titles[cur_pid]

            # 0505 遇到最大的 4只
            # semi - supervised learning by entropy minimization
            # regularization with stochastic transformations and perturbations for deep semi-supervised learning.
            # temporal ensembling for semi - supervised learning.
            # mixup: beyond empirical risk minimization.

            # TODO: 0505的check,看有多少ref_source max
            # if len(source_titles)>=5:
            #     for i in source_titles:
            #         print(i)
            #     assert ()
            if len(source_titles) == 0:
                continue
            # xml论文，再次获取相关信息
            references = bs.find_all("biblStruct")
            # 获取待寻找PST paper的xml所有索引论文
            bid_to_title = {}
            bid_to_publish_year = {}
            bid_to_quote_times = {}
            n_refs = 0
            for ref in references:
                if "xml:id" not in ref.attrs:
                    continue
                bid = ref.attrs["xml:id"]

                # 查看引用次数
                if (bid not in bid_to_quote_times):
                    bid_to_quote_times[bid] = 1
                else:
                    bid_to_quote_times[bid] += 1
                if ref.analytic is None:
                    continue
                if ref.analytic.title is None:
                    continue

                bid_to_title[bid] = ref.analytic.title.text.lower()
                # B0 ,B1
                b_idx = int(bid[1:]) + 1
                if b_idx > n_refs:
                    n_refs = b_idx
            flag = False

            cur_pos_bib = set()
            cur_pos_bib_year = []

            for bid in bid_to_title:
                cur_ref_title = bid_to_title[bid]
                # 可能不止 1篇 source_titles
                for label_title in source_titles:
                    # rf_paper 对比 source_paper 相似度,单纯从文章名字，剔除一部分先
                    # 打印出两个标题之间的相似度分数
                    # 2024-05-05 排除引用不匹配的情况,即错误的源头论文？
                    if fuzz.ratio(cur_ref_title, label_title) >= 80:
                        # singan: learning a generative model from a single natural image
                        # singan: learning a generative model from a single natural image
                        # print(cur_ref_title," ",label_title)
                        flag = True
                        if bid not in cur_pos_bib:
                            cur_pos_bib.add(bid)
                            # cur_pos_bib_year.append(bid_to_publish_year[bid])

            # cur_pos_bib是 于sorce_paper名字相似的rf论文，设置为pos
            # 其余的rf_paper 设置为ng_paper

            cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

            if not flag:
                continue

            if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
                continue
            # 从XML数据中提取带有引用标记的文本上下文
            # 即 获取引用【1】的上下文100 共200字符
            bib_to_contexts = utils.find_bib_context(xml)
            # bib_to_contexts, bib_context_times = utils.find_bib_context(xml)
            # for i in bib_context_times:
            #     if(bib_context_times[i]>1):
            #         print(i)
            #         print(bib_to_contexts[i])
            #         print("---")
            # assert()
            n_pos = len(cur_pos_bib)
            # TODO:0506 因为重复采取，所以会有同样的片段出现，原始负样本数量如下
            n_neg = n_pos * 10
            # n_neg = min (n_pos*10,len((cur_neg_bib)))

            # n_neg = len(cur_neg_bib)

            # 可重复抽取 负样本
            # 负样本是相反
            # 正样本是 title于source_paper相似度大于80

            # replace = true 代表放回抽样，n_neg代表抽取个数
            # TODO:0506  原始采负样本如下
            cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)
            # cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=False)
            # cur_neg_bib_sample = list(cur_neg_bib)

            # cur_neg_bib_year = []
            # 出版年份没啥用，0501已经试过了
            # for ii in cur_neg_bib_sample:
            #     cur_neg_bib_year.append(bid_to_publish_year[ii])

            if cur_pid in pids_train:
                cur_x = x_train
                cur_y = y_train
            elif cur_pid in pids_valid:
                cur_x = x_valid
                cur_y = y_valid
            else:
                continue
                # raise Exception("cur_pid not in train/valid/test")

            orig_title = pid_to_title[cur_pid]
            num = 0
            pos_null_count = 0
            neg_null_count = 0
            for bib in cur_pos_bib:
                # print(bib_to_contexts[bib])
                # assert()
                # for item in bib_to_contexts[bib]:
                #     cur_context = item
                #     cur_x.append(cur_context)
                #     cur_y.append(1)
                #     num += 1
                # TODO: 0524 original is not split ,like below
                # TODO: choose to split
                # TODO:'d ecosystem from desertification.The emergence of indicator plants
                # is an important sign of grassland degradation [Zhao et al., 2004].
                # Many countries have successfully used specific plant species a@@@',
                # 'land would go through five degradation stages before desertification,
                # with the coverage of SC building up in each stage [Zhao et al., 2004]
                # , as shown in Table 1. T@@@',
                # 'a mapping relationship between SC coverage and the degradation stage
                # (Table 1) [Zhao et al., 2004] to obtain the degradation estimation.
                # In this way, we accomplis@@@']

                # ---------------- ori
                cur_title = bid_to_title[bib]
                cur_context = " ".join(bib_to_contexts[bib])
                # cur_x.append(
                #     "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                # cur_y.append(1)
                # num += 1

                # cur_x.append("<p> " + orig_title + " </p><p> " + cur_title + " </p><p> " + cur_context + " </p>")
                # cur_x.append("ori title is " + orig_title + " ref title is " + cur_title + " cite context is" + cur_context)
                # cur_x.append(cur_context)
                if (cur_context):
                    cur_x.append(
                        "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                    cur_y.append(1)
                    num += 1
                    # cur_x.append(cur_context)
                    # cur_x.append("[ori title: " + orig_title + "]" + " [ref title: " + cur_title + "] " + " cita context: " + cur_context)
                    # cur_x.append("[ori title is "+ orig_title + "]"+" [ref title is " + cur_title + "] "+"cita context " + cur_context)
                    # cur_y.append(1)
                    # num += 1
                # else:
                #     all_null_pos += 1
                #     pos_null_count += 1
                #     if pos_null_count <= 2:
                #         cur_x.append(cur_context)
                #         cur_y.append(1)
                #         num += 1
                # ----------------

                # print(cur_context)
                # print(bib)
                # print(bib_context_times[bib])
                # assert()
                # cur_context = "lecture publish year is " + str(cur_pos_bib_year[num]) +" "+cur_context

                # if (bib in bib_context_times):
                # print("1 ", bib_context_times[bib])
                # else:
                # bib_context_times[bib] = 0
                # print("1 ", bib_context_times[bib])
                # print(cur_context)
                # assert()
            num = 0
            for bib in cur_neg_bib_sample:
                # for item in bib_to_contexts[bib]:
                #     cur_context = item
                #     cur_x.append(cur_context)
                #     cur_y.append(0)
                #     num += 1

                # cur_context = "lecture publish year is " + str(cur_neg_bib_year[num]) + " " + cur_context

                # TODO:original is below
                # --------------------
                cur_title = bid_to_title[bib]
                cur_context = " ".join(bib_to_contexts[bib])
                # cur_x.append(
                #     "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                # cur_y.append(0)
                # num += 1
                # cur_x.append("<title> " + orig_title + " </title><title> " + cur_title + " </title><p> " + cur_context + " </p>")
                # cur_x.append("<p> " + orig_title + " </p><p> " + cur_title + " </p><p> " + cur_context + " </p>")
                # cur_x.append("ori title is " + orig_title + " ref title is " + cur_title + " cite context is" + cur_context)
                # cur_x.append(cur_context)

                if (cur_context):
                    # cur_x.append("[ori title: " + orig_title + "]" + " [ref title: " + cur_title + "] " + " cita context: " + cur_context)
                    # cur_x.append("[ori title is "+ orig_title + "]"+" [ref title is " + cur_title + "] "+"cita context " + cur_context)
                    # cur_x.append("bibr title is " + cur_title + " " + cur_context)
                    cur_x.append(
                        "<title> " + orig_title + " </title><title> " + cur_title + " </title><context> " + cur_context + " </context>")
                    cur_y.append(0)
                    num += 1
                    # cur_x.append(cur_context)
                    # cur_y.append(0)
                    # num += 1
                # else:
                #     all_null_neg += 1
                #     neg_null_count += 1
                #     if neg_null_count <= 2:
                #         cur_x.append(cur_context)
                #         cur_y.append(0)
                #         num += 1
                # # --------------------

                # if (bib in bib_context_times):
                #     print("0 ", bib_context_times[bib])
                # else:
                #     bib_context_times[bib] = 0
                #     print("0 ", bib_context_times[bib])
        # len(x_train) 15059 len(x_valid) 8408
        with open(f"./new_data_len_{count}.txt", "w") as f:
            print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid), file=f)
            print("all_null_pos is ", all_null_pos, " all_null_neg is ", all_null_neg, file=f)
            # all_null_pos += 1
        with open(join(data_dir, f"new_bib_context_train_{count}.txt"), "w", encoding="utf-8") as f:
            for line in x_train:
                f.write(line + "\n")

        with open(join(data_dir, f"new_bib_context_valid_{count}.txt"), "w", encoding="utf-8") as f:
            for line in x_valid:
                f.write(line + "\n")

        with open(join(data_dir, f"new_bib_context_train_label_{count}.txt"), "w", encoding="utf-8") as f:
            for line in y_train:
                f.write(str(line) + "\n")

        with open(join(data_dir, f"new_bib_context_valid_label_{count}.txt"), "w", encoding="utf-8") as f:
            for line in y_valid:
                f.write(str(line) + "\n")



def gen_kddcup_test_submission_bert(model_name="scibert"):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "allenai/scibert_scivocab_uncased"
    elif model_name == "sciroberta":
        BERT_MODEL = "allenai/cs_roberta_base"
    elif model_name == "structbert":
        BERT_MODEL = "structbert"
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    label_mask = utils.load_json(data_dir, "Btest_monogr_mask.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    model = RobertaForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model_lxe.bin")))

    model.to(device)
    model.eval()
    # TODO:0510 原是16，但是predict 无所谓
    BATCH_SIZE = 24
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

        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue
        nu_special = 0
        special = []
        contexts_sorted = []
        contexts_sorted = [
            "<title> " + orig_title + " </title><title> " + bid_to_title[bib] + " </title><context> " + " ".join(bib_to_contexts[bib]) + " </context>"
            for bib in bib_sorted]
        # contexts_sorted = ["[ori title: " + orig_title + "]" + " [ref title: " + bid_to_title[bib] + "] " + "cita context: " + " ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        test_features, exceed,less_exceed = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        all_exceed_num = all_exceed_num + exceed

        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                          token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = r[0]
                logits = r[1]

            logits = torch.softmax(logits, dim=-1)
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)

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

    print("all test exceed number is : ", all_exceed_num)
    utils.dump_json(sub_dict, settings.RESULT_DIR, "test_lxe_sciroberta.json")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise ValueError("Please specify 'train' or 'test' as a command-line argument.")
    train_or_test = sys.argv[1]
    if train_or_test == "train":
        print("------------- start training --------------")
        prepare_bert_input()
        train(model_name="sciroberta")
    elif train_or_test == "valid":
        gen_kddcup_valid_submission_bert(model_name="sciroberta")
        # gen_kddcup_test_submission_bert(model_name="sci-roberta")
    elif train_or_test == "test":
        # gen_kddcup_valid_submission_bert(model_name="sci-roberta")
        gen_kddcup_test_submission_bert(model_name="sciroberta")
    else:
        raise ValueError("Invalid argument. Please specify 'train' or 'valid' or 'test'.")

    # Kfold_prepare_bert_input()
    # Kflod_train(model_name="sci-roberta")






# TODO:0604
# bs = 32 is good
# bs=24 lr=5e-6 epoch=10 is better
# epoch = 10 is good
# sci-roberta is good
#  judge 2/3 or 3/4 is good

 # 10 5e-6 bs=16
 # 0.5300587815236285]
 # 0.31712880628696516]
 # 0.4231047137193982]

 # epoch = 10 5e-6 bs=32
 #  0.513898826724901  0.3188938871929574  0.4160234533719319
 #  0.5770621742085209 0.36243 0.42142

 # epoch = 10 5e-6 bs=24
# .5977315519600225, 0.5315075592970001]
#  0.32071736242252563, 0.40860015354476725]
#  0.41279954313841727, 0.40816095982602063]

# Loss history: [0.5400772184662564, 0.5077045874659126, 0.508837562429129, 0.6760074780184842]
# Train AP history: [0.2067863662771283, 0.3070754371503186, 0.4268864674424014, 0.5451944086752875]
# eval AP history: [0.3646274241468246, 0.4156819627358036, 0.42048385928208726, 0.395736902902548


# 0605 <title></title> <context></context>
#Loss history: [0.5460521174958471, 0.5179831314192721, 0.5272932303904077, 0.6564065382082963]
# Train AP history: [0.1959591804731311, 0.3102142384234131, 0.3990718371994463, 0.5070728864739905]
# eval AP history: [0.38162816511062897, 0.42586568264123736, 0.4269026851845637, 0.4239027752629658]

# 0605 BEST <title></title> <context></context> + dynamic dist + pos in context
# hlaf
# bs=24,ep=10_pytorch_model_2___sci-roberta_4503.bin
# Dev loss: 0.536362564863538
# Dev ap: 0.3977745755635318
# Eval ap: 0.45032968044634153

# compare 0605 BEST
# before
# Loss step history: [0.5886593752887828, 0.5865541962829567, 0.504868986603071, 0.5459050887523318]
# Train step AP history: [0.1807755563472686, 0.31595070890336635, 0.39083794439389197, 0.47015141262075233]
# eval step AP history: [0.3244591159680757, 0.41354227119702913, 0.45619464003180626, 0.44081014102105665]


# Loss history: [0.5669623255539852, 0.515453437900847, 0.5662042956063702]
# Train AP history: [0.24118392470172978, 0.36629290727553854, 0.48577889096472715]
# eval AP history: [0.4311192725500374, 0.45754290746464465, 0.4509931543336734]
#
# Loss step history: [0.5676925311422651, 0.5054284155748452, 0.5274001860599609, 0.5401276085691847]
# Train step AP history: [0.18532868630335086, 0.36056360245521857, 0.4665797298688293, 0.5556843998062481]
# eval step AP history: [0.3903166240245353, 0.4506817020397784, 0.46446636394842694, 0.450435760596305]


# pytorch_model_2_199_sci - roberta_0.46446636_44742.bin


# Loss history: [0.5766323529440781, 0.511969938771478, 0.5201630688946822]
# Train AP history: [0.20883731582552706, 0.3399532164310711, 0.428622888639252]
# eval AP history: [0.42279036891569693, 0.43891640221275113, 0.44894184507558077]

# Loss history: [0.5514473011930481, 0.5387010840514234, 0.5499878209510832, 0.6091928539499072]
# Train AP history: [0.21622540266139229, 0.3547334190204793, 0.4075593765537792, 0.5214222560980405]
# eval AP history: [0.4235956447781103, 0.4492735139194066, 0.46622638335082567, 0.45943460834368144]

