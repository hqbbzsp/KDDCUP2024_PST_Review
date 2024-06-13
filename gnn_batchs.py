import torch
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
import pandas as pd
import os
import sys
import torch.nn as nn
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim
from sklearn.metrics import average_precision_score
# from models_DJ import DJ
# from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
from sklearn.metrics import f1_score
# from transformers import get_linear_schedule_with_warmup
# import random
from os.path import abspath, dirname, join
from collections import defaultdict as dd
import utils_gcn as utils
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import csv
from FlagEmbedding import BGEM3FlagModel
from langchain.text_splitter import SpacyTextSplitter
import json
# from cogdl.oag import oagbert

PROJ_DIR = join(abspath(dirname(__file__)))
OUT_DIR = join(PROJ_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = join(PROJ_DIR, "dataset")
os.makedirs(DATA_DIR, exist_ok=True)
RESULT_DIR = join(PROJ_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)


def xml_abstract(bs):
    div_content = bs.find('div').text.strip()
    return div_content


def prepare_train_gcn_input():

    ############ load bge model ################

    # model = FlagModel('/data/zsp/KDD_PST_2341/models/AI-ModelScope/bge-small-en-v1___5',
    #                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    #                   use_fp16=True)
    model = BGEM3FlagModel("BAAI/bge-m3",
                           use_fp16=True)
    # BAAI/bge-m3
    # model = BGEM3FlagModel('/data/zsp/KDD_PST_2341/models/Xorbits/bge-m3',
    #                    use_fp16=True)
    splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap=20)
    splitter_1 = SpacyTextSplitter(chunk_size=300, chunk_overlap=20)

    #############################################

    data_dir = join(PROJ_DIR,"data", "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    # target_dir = "/data/zsp/KDD_PST_2341/paper-source-trace/data/PST/qwen_explain_for_reference_1_10"
    # n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])

    save_path = join(DATA_DIR, "M3_train")
    os.makedirs(save_path, exist_ok=True)

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

    pids_papers = {p["_id"] for p in papers}
    for cur_pid in tqdm(pids_papers):

        features = []
        edges = []
        labels = []
        node_id = 0

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
    
        bib_to_contexts = utils.find_bib_context(xml,splitter)
        
        # content = xml_content(bs)
        paper_abstract= xml_abstract(bs)
        paper_abstract= splitter_1.split_text(paper_abstract)
        # 论文标题加入节点
        orig_title = pid_to_title[cur_pid]
        orig_nodeid = node_id
        node_id += 1
        orig_title_feature = model.encode(orig_title)['dense_vecs']
        features.append(orig_title_feature)
        labels.append([orig_nodeid,-1])
        # 论文摘要加入节点,与论文标题建边
        # abstract_nodes = []
        for acstract_text in paper_abstract:
            acstract_nodeid = node_id
            # abstract_nodes.append(acstract_nodeid)
            node_id += 1
            acstract_feature = model.encode(acstract_text)['dense_vecs']
            features.append(acstract_feature)
            labels.append([acstract_nodeid,-1])
            edges.append([acstract_nodeid,orig_nodeid])

        for bib in bid_to_title.keys():
            b_idx = int(bib[1:]) + 1
            # 参考文献标题加入节点
            cur_title = bid_to_title[bib]
            cur_nodeid = node_id
            cur_title_feature = model.encode(cur_title)['dense_vecs']
            features.append(cur_title_feature)
            if bib in cur_pos_bib:
                labels.append([cur_nodeid,1])
            else :
                labels.append([cur_nodeid,0])
            node_id += 1
            # 标题双向建边
            edges.append([orig_nodeid,cur_nodeid])
            edges.append([cur_nodeid,orig_nodeid])
            # 标题与摘要双向建边
            # for ab_id in abstract_nodes:
            #     edges.append([ab_id,cur_nodeid])
            #     edges.append([cur_nodeid,ab_id])

            # 处理参考文献
            context_list=bib_to_contexts[bib]
            for context in context_list:
                # 创建句子节点
                sentence_nodeids=[]
                # assert len(context) == 3
                sentence_feature = model.encode(context)['dense_vecs']
                # assert len(sentence_features) == 3
                # for sentence_feature in sentence_features:
                    # 句子加入节点
                sentence_nodeids.append(node_id)
                features.append(sentence_feature)
                labels.append([node_id,-1])
                node_id+=1
                # 句子与参考文献双向建边
                for sentce_id in sentence_nodeids:
                    edges.append([sentce_id,cur_nodeid])

        
        f.close()


        if not os.path.exists(f'{save_path}/{cur_pid}'):
            os.mkdir(f'{save_path}/{cur_pid}')

        with open(f'{save_path}/{cur_pid}/features.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(features)
        with open(f'{save_path}/{cur_pid}/edges.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(edges)
        with open(f'{save_path}/{cur_pid}/labels.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(labels)

def prepare_test_gcn_input():

    ############ load bge model ################

    # model = FlagModel('/data/zsp/KDD_PST_2341/models/AI-ModelScope/bge-small-en-v1___5',
    #                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    #                   use_fp16=True)
    # BAAI/bge-m3
    splitter = SpacyTextSplitter(chunk_size=200, chunk_overlap=20)
    model = BGEM3FlagModel("BAAI/bge-m3",
                           use_fp16=True)
    # model = BGEM3FlagModel('/data/zsp/KDD_PST_2341/models/Xorbits/bge-m3',
    #                    use_fp16=True)
    splitter_1 = SpacyTextSplitter(chunk_size=300, chunk_overlap=20)

    #############################################

    data_dir = join(PROJ_DIR,"data", "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    
    xml_dir = join(data_dir, "paper-xml")
    save_path = join(DATA_DIR, "M3_test")
    os.makedirs(save_path, exist_ok=True)
    
    test_mask = dd(list)
    for paper in tqdm(papers):

        features = []
        edges = []
        labels = []
        node_id = 0

        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        orig_title = paper["title"]
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

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

        # if len(sub_example_dict[cur_pid]) != n_refs:
        #     print(cur_pid)
        #     print(len(sub_example_dict[cur_pid]))
        #     print(n_refs)
        assert len(sub_example_dict[cur_pid]) == n_refs
        bib_to_contexts = utils.find_bib_context(xml,splitter)
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        # content = xml_content(bs)
        paper_abstract= xml_abstract(bs)
        paper_abstract= splitter_1.split_text(paper_abstract)
        # 论文标题加入节点
        orig_nodeid = node_id
        node_id += 1
        orig_title_feature = model.encode(orig_title)['dense_vecs']
        features.append(orig_title_feature)
        labels.append([orig_nodeid,-1])
        # 论文摘要加入节点,与论文标题建边
        for acstract_text in paper_abstract:
            acstract_nodeid = node_id
            node_id += 1
            acstract_feature = model.encode(acstract_text)['dense_vecs']
            features.append(acstract_feature)
            labels.append([acstract_nodeid,-1])
            edges.append([acstract_nodeid,orig_nodeid])
        # continue
        # print(bid_to_title.keys())
        # for ind in range(len(sub_example_dict[cur_pid])):
        for bib in bib_sorted:
            context_list=bib_to_contexts[bib]
            # 参考文献标题加入节点
            if bib in bid_to_title.keys():
                test_mask[cur_pid].append(1)
            else :
                test_mask[cur_pid].append(0)
                continue
            cur_title = bid_to_title[bib]
            cur_nodeid = node_id
            cur_title_feature = model.encode(cur_title)['dense_vecs']
            features.append(cur_title_feature)
            labels.append([cur_nodeid,1])
            node_id+=1
            # 标题双向建边
            edges.append([orig_nodeid,cur_nodeid])
            edges.append([cur_nodeid,orig_nodeid])

            # 处理参考文献
            for context in context_list:
                # 创建句子节点
                sentence_nodeids=[]
                # assert len(context) == 3
                sentence_feature = model.encode(context)['dense_vecs']
                # assert len(sentence_features) == 3
                # for sentence_feature in sentence_features:
                    # 句子加入节点
                sentence_nodeids.append(node_id)
                features.append(sentence_feature)
                labels.append([node_id,-1])
                node_id+=1
                for sentce_id in sentence_nodeids:
                    edges.append([sentce_id,cur_nodeid])

        if not os.path.exists(f'{save_path}/{cur_pid}'):
            os.mkdir(f'{save_path}/{cur_pid}')

        with open(f'{save_path}/{cur_pid}/features.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(features)
        with open(f'{save_path}/{cur_pid}/edges.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(edges)
        with open(f'{save_path}/{cur_pid}/labels.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(labels)

    with open(f"{data_dir}/Btest_monogr_mask.json",'w') as f:
        json.dump(test_mask,f,indent=4)

def prepare_data(fold = 0):
    file_path = f"{PROJ_DIR}/dataset/M3_train"
    file_list = os.listdir(file_path)
    print(f"---------------- graph data num : {len(file_list)} ----------------")

    file_list = sorted(file_list)
    n_train = int(len(file_list) * 1 / 3)
    # random.seed(2024)
    # random.shuffle(file_list)

    if fold == 0:
        test_file_list = file_list[:n_train]
        trian_file_list = file_list[n_train:]
    elif fold == 1:
        test_file_list = file_list[n_train:2*n_train]
        trian_file_list = file_list[:n_train]+file_list[2*n_train:]
    else :
        test_file_list = file_list[2*n_train:]
        trian_file_list = file_list[:2*n_train]

    train_data_list = []
    for graph in tqdm(trian_file_list):
        features = pd.read_csv(f'{file_path}/{graph}/features.csv', header=None).values
        features = torch.tensor(features, dtype=torch.float)

        edges = pd.read_csv(f'{file_path}/{graph}/edges.csv', header=None).values
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        labels = pd.read_csv(f'{file_path}/{graph}/labels.csv', header=None)
        node_labels = labels.iloc[:, 1].values
        node_labels = torch.tensor([label if label != -1 else 0 for label in node_labels], dtype=torch.long)
        if 1 not in node_labels:
            print(graph)
            print("not pos")
            continue

        # labeled_nodes = labels[labels.iloc[:, 1] != -1].index
        labeled_nodes = labels[labels.iloc[:, 1] != -1].index
        # abstract_nodes = labels[labels.iloc[:, 1] == -2].index
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[labeled_nodes] = True
        test_mask = torch.zeros(len(labels), dtype=torch.bool)

        # 拓展维度特征
        auxiliary_info = torch.zeros(len(features), 1,  dtype=torch.long)
        auxiliary_info[labeled_nodes] = 1  # 标记需要分类的节点为1
        features = torch.cat([features, auxiliary_info], dim=1)

        data = Data(x=features, edge_index=edge_index, y=node_labels)
        data.train_mask = train_mask
        data.test_mask = test_mask

        train_data_list.append(data)

    test_data_list = []
    for graph in tqdm(test_file_list):
        features = pd.read_csv(f'{file_path}/{graph}/features.csv', header=None).values
        features = torch.tensor(features, dtype=torch.float)

        edges = pd.read_csv(f'{file_path}/{graph}/edges.csv', header=None).values
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        labels = pd.read_csv(f'{file_path}/{graph}/labels.csv', header=None)
        node_labels = labels.iloc[:, 1].values
        node_labels = torch.tensor([label if label != -1 else 0 for label in node_labels], dtype=torch.long)
        if 1 not in node_labels:
            print("not pos")
            continue

        # labeled_nodes = labels[labels.iloc[:, 1] != -1].index
        labeled_nodes = labels[labels.iloc[:, 1] != -1].index
        # abstract_nodes = labels[labels.iloc[:, 1] == -2].index
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[labeled_nodes] = True
        test_mask = torch.zeros(len(labels), dtype=torch.bool)

        # 拓展维度特征
        auxiliary_info = torch.zeros(len(features), 1,  dtype=torch.long)
        auxiliary_info[labeled_nodes] = 1  # 标记需要分类的节点为1
        features = torch.cat([features, auxiliary_info], dim=1)

        data = Data(x=features, edge_index=edge_index, y=node_labels)
        data.train_mask = test_mask
        data.test_mask = train_mask

        test_data_list.append(data)

    return train_data_list,test_data_list


# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_aps = []
    all_f1s = []
    all_losses = []
    for data in loader:
        data = data.to(device)

        fixed_y = data.y
        fixed_train_mask = data.train_mask

        logits = model(data)
        loss = criterion(logits[fixed_train_mask], fixed_y[fixed_train_mask])
        loss = loss
        all_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        logits = torch.softmax(logits, dim=-1)
        y_pre=logits[fixed_train_mask][:, 1].cpu().detach().numpy().tolist()
        y_true=fixed_y[fixed_train_mask].cpu().detach().numpy().tolist()
        eval_ap = average_precision_score(y_true, y_pre)
        all_aps.append(eval_ap)

        pred = logits.argmax(dim=1)[fixed_train_mask]
        label = fixed_y[fixed_train_mask]
        f1 = f1_score(label.cpu(), pred.cpu(), average='weighted')
        all_f1s.append(f1)

    aps = sum(all_aps) / len(all_aps)
    f1s = sum(all_f1s) / len(all_f1s)
    losses = sum(all_losses) / len(all_losses)
            
    return aps, f1s, losses

# 测试函数
def test(model, loader, device, criterion):
    model.eval()
    all_aps = []
    all_f1s = []
    all_losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            fixed_y = data.y
            fixed_test_mask = data.test_mask

            # logits, loss_norm = model(fixed_x, fixed_adj)
            logits= model(data)
            loss = criterion(logits[fixed_test_mask], fixed_y[fixed_test_mask])
            loss =   loss
            all_losses.append(loss.item())

            logits = torch.softmax(logits, dim=-1)
            y_pre=logits[fixed_test_mask][:, 1].cpu().detach().numpy().tolist()
            y_true=fixed_y[fixed_test_mask].cpu().detach().numpy().tolist()
            eval_ap = average_precision_score(y_true, y_pre)
            all_aps.append(eval_ap)

            pred = logits.argmax(dim=1)[fixed_test_mask]
            label = fixed_y[fixed_test_mask]
            f1 = f1_score(label.cpu(), pred.cpu(), average='weighted')
            all_f1s.append(f1)

    aps = sum(all_aps) / len(all_aps)
    f1s = sum(all_f1s) / len(all_f1s)
    losses = sum(all_losses) / len(all_losses)
            
    return aps, f1s, losses




# 创建数据加载器
# batch_size = 1
# train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels,out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    

def train(fold = 0):
    # 获取输入和输出维度
    # first_batch = next(iter(train_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weight = torch.Tensor([0.5,10.0]).to(device)
    train_data_list,test_data_list = prepare_data(fold)
    first_batch = train_data_list[0]
    in_channels = first_batch.num_node_features
    out_channels = len(torch.unique(first_batch.y))
    print(f"num_node_features {in_channels} ")
    print(f"num_node_classes {out_channels} ")
    model = GCNModel(in_channels, 384, out_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    num_epochs = 100
    criterion = torch.nn.NLLLoss(weight=class_weight)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    OUTPUT_DIR = join(OUT_DIR, "kddcup", "gcn")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, f"best_loss_gcnmodel_{fold}.pth")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练模型
    paitence = 0
    best_loss = 100
    print(f"---------------- fold {fold} start training ----------------")
    for epoch in range(num_epochs):
        aps_train, f1s_train, losses_train = train_one_epoch(model, train_data_list, optimizer, criterion, device)
        scheduler.step()
        aps_test, f1s_test, losses_test = test(model, test_data_list, device,criterion)
        # if best_AP < aps_test:
        #     paitence = 0
        #     torch.save(model.state_dict(), model_save_path)
        #     best_AP = aps_test
        # else :
        #     paitence+=1
        if best_loss > losses_test:
            paitence = 0
            torch.save(model.state_dict(), model_save_path)
            best_loss = losses_test
        else :
            paitence+=1
        if paitence == 10:
            break
        print(f'Epoch: {epoch:03d}/{num_epochs:03d}, Train Loss: {losses_train:.4f}, Test Loss: {losses_test:.4f}\t|| Train AP: {aps_train:.4f}, Test AP: {aps_test:.4f}\t|| Train F1: {f1s_train:.4f}, Test F1: {f1s_test:.4f}\t|| Best Test loss: {best_loss:.4f}')

def predict(fold = 0):
    OUTPUT_DIR = join(OUT_DIR, "kddcup", "gcn")
    model_path = os.path.join(OUTPUT_DIR, f"best_loss_gcnmodel_{fold}.pth")
    
    data_dir = join(PROJ_DIR,"data", "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    label_mask = utils.load_json(data_dir, "Btest_monogr_mask.json")
    # test_mask_keys = test_mask.keys()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # model = GCN(385, 2)
    model = GCNModel(1025, 384, 2).to(device)
    model.load_state_dict(torch.load(model_path))

    ########### load data ##############
    file_list = join(DATA_DIR, "M3_test")
    # files = os.listdir(file_list)

    model.to(device)
    model.eval()

    sub_dict = {}
    # all_aps =[]

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        # cur_pid = paper

        graph_path = os.path.join(file_list,cur_pid)

        features = pd.read_csv(f'{graph_path}/features.csv', header=None).values
        features = torch.tensor(features, dtype=torch.float)

        edges = pd.read_csv(f'{graph_path}/edges.csv', header=None).values
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        labels = pd.read_csv(f'{graph_path}/labels.csv', header=None)
        node_labels = labels.iloc[:, 1].values
        node_labels = torch.tensor([label if label != -1 else 0 for label in node_labels], dtype=torch.long)

        labeled_nodes = labels[labels.iloc[:, 1] != -1].index
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[labeled_nodes] = True
        test_mask = torch.zeros(len(labels), dtype=torch.bool)

        # 拓展维度特征
        auxiliary_info = torch.zeros(len(features), 1,  dtype=torch.long)
        auxiliary_info[labeled_nodes] = 1  # 标记需要分类的节点为1
        features = torch.cat([features, auxiliary_info], dim=1)

        data = Data(x=features, edge_index=edge_index, y=node_labels)
        data.train_mask = train_mask
        data.test_mask = test_mask

        n_refs=len(labeled_nodes)

        # assert len(sub_example_dict[cur_pid]) == n_refs
        # continue

        data = data.to(device)
        logits= model(data)
        logits = torch.softmax(logits, dim=-1)
        y_pre=logits[train_mask][:, 1]
        y_pres = torch.zeros(len(sub_example_dict[cur_pid]), dtype=torch.float).to(device)
        # mask_test = np.array(label_mask[cur_pid])
        mask = label_mask[cur_pid]
        labeled_index = [i for i,itm in enumerate(mask) if itm == 1]
        # labeled_index = torch.tensor(labeled_index,dtype=torch.long).to(device)
        y_pres[labeled_index]=y_pre
        y_pres = y_pres.cpu().detach().numpy().tolist()
        sub_dict[cur_pid] = y_pres
        # print(y_score)
        # y_true=data.y[train_mask].cpu().detach().numpy().tolist()
        # print(torch.bincount(fixed_y[fixed_test_mask]))
        # eval_ap = average_precision_score(y_true, y_pre)
        # all_aps.append(eval_ap)
        # print(eval_ap)
    # print(sum(all_aps) / len(all_aps))
    
    utils.dump_json(sub_dict, RESULT_DIR, f"gcn_fold{fold}.json")

def merge_result():
    data_dir = RESULT_DIR
    data_1 = utils.load_json(data_dir, "gcn_fold0.json")
    data_2 = utils.load_json(data_dir, "gcn_fold1.json")
    data_3 = utils.load_json(data_dir, "gcn_fold2.json")
    masks_mo = utils.load_json(join(PROJ_DIR,"data", "PST"),"Btest_monogr_mask.json")

    result={}
    lis = data_1.keys()

    for key in lis:
        item_1 = data_1[key]
        item_2 = data_2[key]
        item_3 = data_3[key]
        mask_mo = masks_mo[key]
        result_1 = []
        for idx in range(len(item_1)):
            if mask_mo[idx]:
                result_1.append(item_1[idx]*0.3+item_2[idx]*0.3+item_3[idx]*0.4)
            else :
                result_1.append(0.0)
        result[key]=result_1
    utils.dump_json(result, RESULT_DIR, "gcn_fold3.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please specify 'train' or 'test' as a command-line argument.")

    train_or_test = sys.argv[1]
    if train_or_test == "train":
        prepare_train_gcn_input()
        for i in range(3):
            train(i)
    elif train_or_test == "test":
        prepare_test_gcn_input()
        for i in range(3):
            predict(i)
        merge_result()
    else:
        raise ValueError("Invalid argument. Please specify 'train' or 'test'.")

