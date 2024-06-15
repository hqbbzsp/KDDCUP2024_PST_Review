# KDDCUP2024_PST_Review

## 2024-6-15-Debug

在Data Prepare中我们提供的检查点为实验此仓库一次性得出的，我们以往提交的检查点可以在[这里](https://pan.baidu.com/s/1c7PfOWbcVdctiVD0loTbEw?pwd=sirg)找到，提取码为```sirg```。但由于我们进行 sciroberta 推理时，transformers库版本不一致，**请分两步执行```test.sh``` 文件**，第一步执行第一行```python gnn_batchs.py test```，第二步切换transformers库版本为```4.22.2```，然后执行```test.sh``` 文件除第一行之外的命令。

非常抱歉，由于文件繁多，某个检查点已经丢失，但结果相差不多，我们能够保证我们方案的真实性。

## Step1 Installation

```
pip install -r requirements.txt
```

Note that spacy uses the ```en_core_web_sm``` model.

If you need to download BGEM3 pre-trained models manually, download the [model](https://huggingface.co/BAAI/bge-m3) location from huggingface to ```/models```, and change line 51 and line 223 of the file ```gnn_batchs.py``` to the path where the model is located.

## Step2 Data Prapare

The training set and validation set can be downloaded from [BaiduPan](https://pan.baidu.com/s/1zylNX4Ar5nZAjNx5mcxSmg?pwd=wzud).

The test set can be downloaded from [BaiduPan](https://pan.baidu.com/s/1CYCW_COrUmuYGI3k_eg7wA?pwd=7f9i).

After downloading, the folder structure should be as follows:

```
data/

├── PST/

│ ├── paper-xml/

│ │ └── XXXX.xml

│ ├── paper_source_gen_by_rule.json

│ ├── paper_source_trace_test_wo_ans.json

│ ├── paper_source_trace_train_ans.json

│ ├── paper_source_trace_valid_wo_ans.json

│ ├── submission_example_test.json

│ └── submission_example_valid.json
```

## Step3 Run the train.sh to reproduce the training model

```
bash train.sh
```

you can get our finetune models from [BaiduPan](https://pan.baidu.com/s/1eCJ4g13x5GAyknmTAZm7ow?pwd=y982). password is : y982

after run train.sh , we will get finetuning models , the folder structure should be as follows: 

```
out/

├── kddcup/

│ ├── gcn/

│ │ ├── best_loss_gcnmodel_0.pth

│ │ ├── best_loss_gcnmodel_1.pth

│ │ └── best_loss_gcnmodel_2.pth

│ ├── sciroberta/

│ │ ├── latest_model.bin

│ │ ├── pytorch_model_0.bin

│ │ ├── pytorch_model_1.bin

│ │ ├── pytorch_model_ab_div_1.bin

│ │ └── pytorch_model_lxe.bin

```



## Step4 Run the test.sh to reproduce the answer of paper-source-trace

```
bash test.sh
```

after run test.sh

we will get result in result folder like below:

```
result/

│ ├── gcn_fold0.json

│ ├── gcn_fold1.json

│ ├── gcn_fold2.json

│ ├── gcn_fold3.json

│ ├── roberta_spacy_fold2.json

│ ├── test_ab_div_sciroberta.json

│ ├── test_fold0_sciroberta.json

│ ├── test_fold1_sciroberta.json

│ ├── test_lxe_sciroberta.json

│ └── test_submission.json
```

the test_submission.json is the final ensemble answer.



if you have any question , please feel free to contact at zspnjlgdx@gmail.com 
