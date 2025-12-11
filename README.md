# AdvancedML-DNN-Pruning

/pruners contains our Pruners.py, a set of Pruning classes to Prune other models.
/Models contain our implementations of ResNet32 (with training/plotting main method), VGG-16, and VIT, as well as an alternate resnet32 implementation that was not used.
post-train-prune.py is our workflow for running PAT experiments
prune.py contains methods to apply Prunering classes to models
train.py Contains an example training function
