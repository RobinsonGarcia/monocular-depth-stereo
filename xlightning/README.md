inital pl version 1.6.5


model's output: disparity space
compute scale and shift for each model
given scale and shift
losses: invert_valid(scale * d  + shift)
metrics:

## preprocessing:
- implement nyu hole filling and compare ploting same samples with and without (both, img and depthmap)


## verifications:

datamodule test 1
1. print 20 by 20 image samples
2. check output with and without disparity

model test
1. test_step , valid_step: compute metrics for a single image
2. check output magnitude's
3. compute scale and shif if necessary


optimization test
training - overfit on a single img, plot loss