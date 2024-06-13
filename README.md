# KDDCUP2024_PST_Review



## Step1 Installation

```
pip install -r requirements.txt
```

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
