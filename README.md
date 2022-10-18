# GIFT: Graph-guIded-Feature-Transfer-Network (CIKM22)


This is the source code for the CIKM 22 paper "[GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction](https://arxiv.org/pdf/2202.11525.pdf)"

## Method

### Graph Construction for Video Feature Transfer
####  Physical Linkage
![image text](https://github.com/Bayi-Hu/GIFT-Graph-guided-Feature-Transfer-Network/blob/master/materials/physical_linkages.png)
#### Semantic Linkage
![image text](https://github.com/Bayi-Hu/GIFT-Graph-guided-Feature-Transfer-Network/blob/master/materials/semantic_linkage.png)

### Graph-guIded Feature Transfer (GIFT) network
![image text](https://github.com/Bayi-Hu/GIFT-Graph-guided-Feature-Transfer-Network/blob/master/materials/GIFT.png)

## Results

We report the results on a large scale video recommendation dataset collected from Guess You Like (猜你喜欢) of Taobao App's homepage. Online evaluation shows that GIFT has brought 6.82% lift on CTR metric (from 4.180% to 4.465% during Sep.21 ~ Sep.27, 2020).

| Model | AUC|
| ------ | ------ |
|DNN|0.7423|
|DeepFM|0.7508|
| DIN  |0.7568 | 
| GIFT |0.7670|
| GIFT with finetune |0.7693|

---
Due to Alibaba Group's privacy policy, we cannot publish the source dataset used in our paper, but we re-implement the GIFT network and conduct experiments on the DBook dataset, which is collected from www.douban.com.

## Getting Start

### Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Preprocess dataset 

* Step 1: Construct physical linkages between new books and old books.
```sh
cd FeatGeneration;
python DBook_graph_construction.py;
python DBook_data_process.py;
```
or 
```sh
sh process_dbook.sh;
```

* Step 2: Train a GIFT (based on DNN) model
```sh
cd Train;
python train_dnn.py;
python eval_dnn.py
``` 

* Step 3: Train and evaluate a DNN model
```sh
cd Train;
python train_dnn_gift.py;
python eval_dnn_gift.py;
``` 

### Results

| Model                     | AUC    |
|---------------------------|--------|
| DNN                       | 0.7103 |
| GIFT (with dot attention) | 0.7175 |
 | GIFT (with mlp attention) | 0.7199 |


---
## Citation

**Slides**

Here is our slides:
https://github.com/Bayi-Hu/GIFT-Graph-guided-Feature-Transfer-Network/blob/master/GIFT_CIKM22_slides.pdf

**Bibtex:**
```
    @article{hu2022gift,
      title={GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction},
      author={Hu, Sihao and Cao, Yi and Gong, Yu and Li, Zhao and Yang, Yazheng and Liu, Qingwen and Ou, Wengwu and Ji, Shouling},
      journal={arXiv preprint arXiv:2202.11525},
      year={2022}
    }
```