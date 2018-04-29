# MemN2N_Experiments
Class project for DS-GA 1008 at NYU to investigate ways to improve MemN2N performance on bAbI tasks. 
Baseline code is from https://github.com/nmhkahn/MemN2N-pytorch. 

**Test accuracy**

#### Baseline results:

mean, std, min
0.33420000000000005 0.06544585548375084 0.236

#### Ensemble with regularization:
ensemble accuracy: 0.453

mean, std, min
0.35007999999999995 0.05676172654174642 0.246

#### Ensemble with regularization and squared cross entropy loss
ensemble_accuracy_so_far 0.443

mean, std, min
0.34629000000000004 0.061368443845351 0.247

