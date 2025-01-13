# Relaxed Rotational Equivariance via $G$-Biases in Vision
The AAAI2025 accepted this paper! We will release the code in the next few days.

**Note:** Unlike the experiments reported in the paper, we simplified the classification header and re-conducted the experiments, resulting in more prominent results with fewer parameters. The new results are as follows:
### Experiments on the CIFAR-100 / 10 dataset
|Model Name|Group $G$|Equivariance|CIFAR-100 Acc.|#Param.|CIFAR-10 Acc.|#Param.|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|c2_sre_n|$C_2$|SRE|76.95|325748|94.68|314138|
|c2_rre_n|$C_2$|RRE|77.84|345140|-|333530|
|c4_sre_n|$C_4$|SRE|80.79|625012|-|-|
|c4_rre_n|$C_4$|RRE|82.48|663796|-|-|
|c6_sre_n|$C_6$|SRE|81.36|924276|-|-|
|c6_rre_n|$C_6$|RRE|83.20|982452|-|-|
|c8_sre_n|$C_8$|SRE|82.52|1223540|-|-|
|c8_rre_n|$C_8$|RRE|83.63|1301108|-|-|

|Model Name|Group $G$|CIFAR-100 Acc.|#Param.|CIFAR-10 Acc.|#Param.|
|:---:|:---:|:---:|:---:|:---:|:---:|
|c4_rre_n|$C_4$|82.48|663796|-|-|
|c4_rre_s|$C_4$|84.47|2524548|-|-|
|c4_rre_m|$C_4$|85.35|8061316|-|-|



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

