# GIFT-Graph-guided-Feature-Transfer-Network

### Introduction

This is the source code for the paper "[GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction](https://arxiv.org/abs/2202.11525)"

### Results

We report the results on a large scale video recommendation dataset collected from Guess You Like (猜你喜欢) of Taobao App's homepage. Online evaluation shows that GIFT has brought 6.82% lift on CTR metric (from 4.180% to 4.465%).

| Model | AUC|
| ------ | ------ |
|DNN|0.7423|
|DeepFM|0.7508|
| DIN  |0.7568 | 
| GIFT |0.7670|
| GIFT with finetune |0.7693|



**Bibtex:**
```
    @article{hu2022gift,
      title={GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction},
      author={Hu, Sihao and Cao, Yi and Gong, Yu and Li, Zhao and Yang, Yazheng and Liu, Qingwen and Ou, Wengwu and Ji, Shouling},
      journal={arXiv preprint arXiv:2202.11525},
      year={2022}
    }
```

## Getting Start

### Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Download dataset and preprocess 

* Step 1: Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
```sh
mkdir raw_data/;
cd utils;
bash 0_download_raw.sh;
```
* Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```sh
python 1_convert_pd.py;
python 2_remap_id.py
```

### Training and Evaluation

This implementation not only contains the GIFT method, but also provides other competitors' methods, including DNN and DIN. The training procedures of all method is as follows:




