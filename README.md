# CIFAR-10 Image Classification

This is a deep learning project for image classification using the CIFAR-10 dataset. It includes a convolutional neural network (CNN) model, training scripts, and prediction tools.

## Introduction

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to train a CNN to classify these images into the correct categories. This README provides an overview of the project and how to use it.

This code does not work because the format of the code got messed up as I was trying to upload it to GitHub as a project from the assignment I was working on.

## Results

```txt
Train Epoch: 0 [12800/50000] Loss: 2.299964 Acc: 0.0938
Train Epoch: 0 [25600/50000] Loss: 2.293073 Acc: 0.1094
Train Epoch: 0 [38400/50000] Loss: 2.248284 Acc: 0.1562
Elapsed 43.57s, 43.57 s/epoch, 0.11 s/batch, ets 827.74s

Test set: Average loss: 2.0347, Accuracy: 2553/10000 (26%)

Accuracy improved, saving the model.

Train Epoch: 1 [12800/50000] Loss: 1.919553 Acc: 0.3438
Train Epoch: 1 [25600/50000] Loss: 1.968104 Acc: 0.3125
Train Epoch: 1 [38400/50000] Loss: 1.682881 Acc: 0.3984
Elapsed 114.26s, 57.13 s/epoch, 0.15 s/batch, ets 1028.32s

Test set: Average loss: 1.8406, Accuracy: 3209/10000 (32%)

Accuracy improved, saving the model.

Train Epoch: 2 [12800/50000] Loss: 1.594030 Acc: 0.3984
Train Epoch: 2 [25600/50000] Loss: 1.685454 Acc: 0.3516
Train Epoch: 2 [38400/50000] Loss: 1.749381 Acc: 0.4375
Elapsed 186.13s, 62.04 s/epoch, 0.16 s/batch, ets 1054.75s

Test set: Average loss: 1.5353, Accuracy: 4434/10000 (44%)

Accuracy improved, saving the model.

Train Epoch: 3 [12800/50000] Loss: 1.618942 Acc: 0.4062
Train Epoch: 3 [25600/50000] Loss: 1.492500 Acc: 0.4844
Train Epoch: 3 [38400/50000] Loss: 1.631959 Acc: 0.4453
Elapsed 257.19s, 64.30 s/epoch, 0.16 s/batch, ets 1028.77s

Test set: Average loss: 1.3449, Accuracy: 5104/10000 (51%)

Accuracy improved, saving the model.

Train Epoch: 4 [12800/50000] Loss: 1.324000 Acc: 0.4766
Train Epoch: 4 [25600/50000] Loss: 1.312502 Acc: 0.5625
Train Epoch: 4 [38400/50000] Loss: 1.383715 Acc: 0.4688
Elapsed 327.94s, 65.59 s/epoch, 0.17 s/batch, ets 983.83s

Test set: Average loss: 1.2713, Accuracy: 5467/10000 (55%)

Accuracy improved, saving the model.

Train Epoch: 5 [12800/50000] Loss: 1.516712 Acc: 0.4766
Train Epoch: 5 [25600/50000] Loss: 1.250219 Acc: 0.5547
Train Epoch: 5 [38400/50000] Loss: 1.178171 Acc: 0.5859
Elapsed 399.49s, 66.58 s/epoch, 0.17 s/batch, ets 932.14s

Test set: Average loss: 1.2061, Accuracy: 5724/10000 (57%)

Accuracy improved, saving the model.

Train Epoch: 6 [12800/50000] Loss: 1.150736 Acc: 0.6016
Train Epoch: 6 [25600/50000] Loss: 1.120121 Acc: 0.6172
Train Epoch: 6 [38400/50000] Loss: 1.165145 Acc: 0.5859
Elapsed 470.57s, 67.22 s/epoch, 0.17 s/batch, ets 873.91s

Test set: Average loss: 1.2569, Accuracy: 5461/10000 (55%)

Train Epoch: 7 [12800/50000] Loss: 1.215543 Acc: 0.5703
Train Epoch: 7 [25600/50000] Loss: 1.128622 Acc: 0.5781
Train Epoch: 7 [38400/50000] Loss: 1.232290 Acc: 0.5156
Elapsed 541.69s, 67.71 s/epoch, 0.17 s/batch, ets 812.53s

Test set: Average loss: 1.0404, Accuracy: 6290/10000 (63%)

Accuracy improved, saving the model.

Train Epoch: 8 [12800/50000] Loss: 0.988673 Acc: 0.6328
Train Epoch: 8 [25600/50000] Loss: 1.094149 Acc: 0.6016
Train Epoch: 8 [38400/50000] Loss: 0.980863 Acc: 0.6953
Elapsed 614.38s, 68.26 s/epoch, 0.17 s/batch, ets 750.91s

Test set: Average loss: 1.0809, Accuracy: 6061/10000 (61%)

Train Epoch: 9 [12800/50000] Loss: 0.853405 Acc: 0.7109
Train Epoch: 9 [25600/50000] Loss: 0.989875 Acc: 0.6328
Train Epoch: 9 [38400/50000] Loss: 0.864294 Acc: 0.7109
Elapsed 684.31s, 68.43 s/epoch, 0.18 s/batch, ets 684.31s

Test set: Average loss: 1.0805, Accuracy: 6154/10000 (62%)

Train Epoch: 10 [12800/50000] Loss: 0.880081 Acc: 0.7344
Train Epoch: 10 [25600/50000] Loss: 0.934914 Acc: 0.6719
Train Epoch: 10 [38400/50000] Loss: 0.897727 Acc: 0.7422
Elapsed 753.83s, 68.53 s/epoch, 0.18 s/batch, ets 616.77s

Test set: Average loss: 0.9429, Accuracy: 6687/10000 (67%)

Accuracy improved, saving the model.

Train Epoch: 11 [12800/50000] Loss: 0.764463 Acc: 0.6875
Train Epoch: 11 [25600/50000] Loss: 0.861323 Acc: 0.7422
Train Epoch: 11 [38400/50000] Loss: 0.825930 Acc: 0.7109
Elapsed 823.71s, 68.64 s/epoch, 0.18 s/batch, ets 549.14s

Test set: Average loss: 0.8882, Accuracy: 6895/10000 (69%)

Accuracy improved, saving the model.

Train Epoch: 12 [12800/50000] Loss: 0.859326 Acc: 0.6875
Train Epoch: 12 [25600/50000] Loss: 0.630371 Acc: 0.7266
Train Epoch: 12 [38400/50000] Loss: 0.783371 Acc: 0.7344
Elapsed 893.58s, 68.74 s/epoch, 0.18 s/batch, ets 481.16s

Test set: Average loss: 0.8429, Accuracy: 7029/10000 (70%)

Accuracy improved, saving the model.

Train Epoch: 13 [12800/50000] Loss: 0.745602 Acc: 0.7266
Train Epoch: 13 [25600/50000] Loss: 0.666629 Acc: 0.7422
Train Epoch: 13 [38400/50000] Loss: 0.704477 Acc: 0.7500
Elapsed 963.93s, 68.85 s/epoch, 0.18 s/batch, ets 413.11s

Test set: Average loss: 0.9106, Accuracy: 6889/10000 (69%)

Train Epoch: 14 [12800/50000] Loss: 0.637790 Acc: 0.7812
Train Epoch: 14 [25600/50000] Loss: 0.723249 Acc: 0.7578
Train Epoch: 14 [38400/50000] Loss: 0.659222 Acc: 0.7734
Elapsed 1033.52s, 68.90 s/epoch, 0.18 s/batch, ets 344.51s

Test set: Average loss: 0.7536, Accuracy: 7428/10000 (74%)

Accuracy improved, saving the model.

Train Epoch: 15 [12800/50000] Loss: 0.642856 Acc: 0.7656
Train Epoch: 15 [25600/50000] Loss: 0.563830 Acc: 0.8125
Train Epoch: 15 [38400/50000] Loss: 0.594618 Acc: 0.7891
Elapsed 1103.30s, 68.96 s/epoch, 0.18 s/batch, ets 275.82s

Test set: Average loss: 0.9416, Accuracy: 6958/10000 (70%)

Train Epoch: 16 [12800/50000] Loss: 0.783305 Acc: 0.7031
Train Epoch: 16 [25600/50000] Loss: 0.558776 Acc: 0.8047
Train Epoch: 16 [38400/50000] Loss: 0.442711 Acc: 0.8359
Elapsed 1172.54s, 68.97 s/epoch, 0.18 s/batch, ets 206.92s

Test set: Average loss: 0.8655, Accuracy: 7163/10000 (72%)

Train Epoch: 17 [12800/50000] Loss: 0.499042 Acc: 0.8359
Train Epoch: 17 [25600/50000] Loss: 0.485057 Acc: 0.8281
Train Epoch: 17 [38400/50000] Loss: 0.417473 Acc: 0.8672
Elapsed 1241.85s, 68.99 s/epoch, 0.18 s/batch, ets 137.98s

Test set: Average loss: 0.7363, Accuracy: 7488/10000 (75%)

Accuracy improved, saving the model.

Train Epoch: 18 [12800/50000] Loss: 0.585107 Acc: 0.7656
Train Epoch: 18 [25600/50000] Loss: 0.448052 Acc: 0.8359
Train Epoch: 18 [38400/50000] Loss: 0.527410 Acc: 0.8125
Elapsed 1312.32s, 69.07 s/epoch, 0.18 s/batch, ets 69.07s

Test set: Average loss: 0.7538, Accuracy: 7490/10000 (75%)

Accuracy improved, saving the model.

Train Epoch: 19 [12800/50000] Loss: 0.410821 Acc: 0.8594
Train Epoch: 19 [25600/50000] Loss: 0.459434 Acc: 0.8438
Train Epoch: 19 [38400/50000] Loss: 0.388209 Acc: 0.8594
Elapsed 1384.19s, 69.21 s/epoch, 0.18 s/batch, ets 0.00s

Test set: Average loss: 0.8128, Accuracy: 7399/10000 (74%)

Total time: 1412.19, Best Loss: 0.736, Best Accuracy: 0.749
```
