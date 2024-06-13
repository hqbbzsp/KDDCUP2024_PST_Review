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
you can get models from [BaiduPan](https://pan.baidu.com/s/1eCJ4g13x5GAyknmTAZm7ow?pwd=y982). password is : y982

## Step4 Run the test.sh to reproduce the answer of paper-source-trace
```
bash test.sh
```
