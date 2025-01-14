# Relaxed Rotational Equivariance via $G$-Biases in Vision
The AAAI2025 accepted this paper! We will release the code in the next few days.

## Classification 
### Training settings
batch_size=128; epochs=300; warmup_epochs=20; lr=0.1; lr_min=1e-6; momentum=0.9; weight_decay=5e-4
augmentation=cutmix_or_mixup (torchvision.transforms.v2); optimizer=SGD; criterion=CrossEntropyLoss;
warmup stage: LinearLR; main_stage: CosineAnnealingLR

**Note:** Unlike the experiments reported in the paper, we simplified the classification header and re-conducted the experiments, resulting in more prominent results with fewer parameters. The new results are as follows:
### Experiments on the CIFAR-100 / 10 dataset
|Model Name|Group $G$|Equivariance|CIFAR-100 Acc.|#Param.|CIFAR-10 Acc.|#Param.|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|c2_sre_n|$C_2$|SRE|76.95|325748|94.68|314138|
|c2_rre_n|$C_2$|RRE|77.84|345140|94.80|333530|
|c4_sre_n|$C_4$|SRE|80.79|625012|95.96|613402|
|c4_rre_n|$C_4$|RRE|82.48|663796|96.62|652186|
|c6_sre_n|$C_6$|SRE|81.36|924276|96.07|912666|
|c6_rre_n|$C_6$|RRE|83.20|982452|96.86|970842|
|c8_sre_n|$C_8$|SRE|82.52|1223540|96.34|1211930|
|c8_rre_n|$C_8$|RRE|83.63|1301108|97.18|1289498|

|Model Name|Group $G$|CIFAR-100 Acc.|#Param.|CIFAR-10 Acc.|#Param.|
|:---:|:---:|:---:|:---:|:---:|:---:|
|c4_rre_n|$C_4$|82.48|663796|96.62|652186|
|c4_rre_s|$C_4$|84.47|2524548|96.90|2501418|
|c4_rre_m|$C_4$|85.35|8061316|97.14|8026666|



You can train the model using the following command:

```
python classification/train.py -model [Model Name] -dataset [Dataset] -batch_size 128
```

**[Model Name]**: c2_sre_n, c2_rre_n, c4_sre_n, c4_rre_n, c4_rre_s, c4_rre_m, c6_sre_n, c6_rre_n, c8_sre_n, c8_rre_n

**[Dataset]**: cifar10, cifar100

Example command: 

```
cd classification
python train.py -model c4_rre_n -dataset cifar100 -batch_size 128
```

## Object detection
### Experiments on the PASCAL VOC07+12 dataset

|Model|AP50|AP50:90|#Param.|
|:---:|:---:|:---:|:---:|
|RREDet-n ($C_4$)|84.1|65.2|2.9M|
|RREDet-s ($C_4$)|86.3|67.6|10.5M|
|RREDet-m ($C_4$)|87.4|70.3|25.7M|


