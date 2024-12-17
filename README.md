# Relaxed Rotational Equivariance via G-Biases in Vision
The AAAI2025 accepted this paper! We will release the code in the next few days.
### Ablation Experiments on the CIFAR-100 dataset
Unlike the ablation experiment reported in the paper, we simplified the classification header and re-conducted the ablation experiment, resulting in more prominent results. The results are as follows:
|Model Name|Group G|Type of Equivariance|Top-1 Acc.|#Param.|
|-|-|-|-|-|
|c2_sre_n|$C_2$|SRE|76.95|-|
|c2_rre_n|$C_2$|RRE|77.84|-|
|c4_sre_n|$C_4$|SRE|80.79|-|
|c4_rre_n|$C_4$|RRE|82.48|-|
|c6_sre_n|$C_6$|SRE|81.36|-|
|c6_rre_n|$C_6$|RRE|83.20|-|
|c8_sre_n|$C_8$|SRE|82.52|-|
|c28_rre_n|$C_8$|RRE|83.63|-|
