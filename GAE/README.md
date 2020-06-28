# The implement of [Variational Graph Auto-Encoders](https://arxiv.org/pdf/1611.07308.pdf)

## 1. Data format
### 1.1 Cora
|file|shape|content|
|:---:|:---:|:---:|
|ind.cora.x|(140, 1442)|?| 
|ind.cora.tx|(1000, 1433)|feature vectors of test samples| 
|ind.cora.allx|(1708, 1433)|feature vectors of train samples|
|ind.cora.graph|(2708, 2708)|adjacency matrix for links|
|ind.cora.text.index|(1000, 1)|indexs of test samples on graph|

## 2. Performance
### 2.1 performance on Cora
|method|AUC|AP|
|:---:|:---:|:---:|
|GAE|0.9047|0.91322|
|VGAE|||

## 3. Ablation study
### 3.1 Influence of "pos_weight" (postive weight) used in loss computation
|model|pos_weight|AUC|AP|
|:---:|:---:|:---:|:---:|
|GAE|original value (about 815)|0.9047|0.91322|
|GAE|1.0|0.5047|0.5047|
|GAE|50.0|0.5476|0.5482|
|GAE|200.0|0.9202|0.9336|
|GAE|400.0|0.9188|0.9301|
|GAE|800.0|0.9146|0.9290|    
