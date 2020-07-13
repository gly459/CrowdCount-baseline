# CrowdCount-baseline

the architecture of the code is the same as [CSRNet](https://arxiv.org/abs/1802.10062), which is the official implementation of the paper [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062).

### train
try `python train.py part_A_train.json part_A_test.py 0 b_` to start the training process

### model
the architecture of the model is the CSRNet A, VGG16 backbone with the dilation rate 1 of the decoder.

### result
Shanghai Tech part A

MAE:67.906  MSE:117.762
| model         | MAE  | MSE  |
| ------------- | ---- | ---- |
| VGG           |67.906|117.762|
| VGG in CSRNet |69.7|116.0|
| VGG*          |65.721|109.128|
| VGG* in ADSCNet|66.5|106.9|
ps:here, I don't split the validation set from the training set and just use the test set, so it's normal the result is better.
