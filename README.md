# Codebase for "DPBag"

Authors: James Jordon, Jinsung Yoon, Mihaela van der Schaar

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate", 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate

Contact: jsyoon0823@gmail.com

This directory contains implementations of DPBag framework for constructing differentially private classification model
using a real-world dataset.

-   Adult data: https://archive.ics.uci.edu/ml/datasets/adult

To run the pipeline for training and evaluation on DPBag framwork, simply run 
python3 -m main_dpbag.py.


### Code explanation

(1) data_loading.py
- Transform raw data to preprocessed data (train, test, valid sets)

(2) dpbag.py
- Core Differentially Private Bagging algorithm
- Use train and valid sets with user-defined parameters (n, k, epsilon, delta) to construct differentially private classification model  
- Use test set to evaluate the performance of differentially private classification model

(3) main_dpbag.py
- Report Accuracy, AUC, APR, and Privacy Budget for each differential privacy inputs


### Command inputs:

-   data_name: adult
-   epsilon: differential privacy parameter
-   delta: differential privacy parameter
-   teacher_no: the number of teachers in PATE framework
-   lamda: hyper-parameter of DPBag on DP noise
-   part_no: the number of partitions
-   exp_iteration: the number of experiment iterations

Note that network parameters should be optimized.

### Example command

```shell
$ python3 main_dpbag.py --data_name adult --epsilon 5 --delta 0.001
--teacher_no 100 --lamda 0.02 --part_no 50 --exp_iteration 5
```

### Outputs

-   metric_results: accuracy, area under roc curve, average precision recall, privacy budget