"""DPBag Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate", 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate

Last updated Date: April 26th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_dpbag.py

(1) Import data
(2) Train DPBag
(3) Evaluate the performances of DPBag
  - Accuracy
  - AUC
  - APR
  - Privacy budget
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 1. DPBag model
from dpbag import dpbag
# 2. Data loading
from data_loading import data_loading_adult


def main (args):
  """Main function for DPBag experiments.
  
  Args:
    - data_name: adult
    - Differential privacy parameters
      - epsilon
      - delta    
    - DPBat parameters (should be optimized for different datasets)
      - teacher_no: the number of teachers in PATE framework
      - lamda: hyper-parameter of DPBag on DP noise
      - part_no: the number of partitions
    - exp_iteration: the number of experiment iterations
  
  Returns:
    - metric_results: performance on testing data
      - acc: accuracy
      - auc: area under roc curve
      - apr: average precision recall
      - budget: privacy budget (the number of samples to be seen by DPBag)
  """        
  # Algorithm parameters
  parameters = dict()  
  parameters['epsilon'] = args.epsilon
  parameters['delta'] = args.delta
  parameters['teacher_no'] = args.teacher_no
  parameters['lamda'] = args.lamda
  parameters['part_no'] = args.part_no
    
  ## Output initializations
  out_acc = list()
  out_auc = list()
  out_apr = list()
  out_budget = list()

  ## Iterate DPBag experiments
  for itr in tqdm(range(args.exp_iteration)):
    
    # Data loading
    train_x, train_y, valid_x, valid_y, test_x, test_y = data_loading_adult()
    print(args.data_name + ' dataset is ready.')
    
    # DPBag train / evaluate
    temp_acc, temp_auc, temp_apr, temp_budget = \
    dpbag(train_x, train_y, valid_x, test_x, test_y, parameters)    
    print('Finish DPBag train/test for ' + str(itr+1) + '-th iteration')
        
    # Stack performances of each experiment
    out_acc.append(temp_acc)
    out_auc.append(temp_auc)
    out_apr.append(temp_apr)
    out_budget.append(temp_budget)
        
  ## Performance Table          
  dict_metrics = {'Epsilon':[i+1 for i in range(len(out_acc[0]))],
                  'Delta': args.delta,
                  'Accuracy': np.mean(out_acc,0),
                  'AUC': np.mean(out_auc,0),
                  'APR': np.mean(out_apr,0),
                  'Budget': np.mean(out_budget,0)}
  
  metric_results = pd.DataFrame(dict_metrics)
  
  ## Print performance metrics on testing data
  print(metric_results)

  return metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['adult'],
      default='adult',
      type=str)
  parser.add_argument(
      '--epsilon',
      help='differential privacy parameter',
      default=5,
      type=int)
  parser.add_argument(
      '--delta',
      help='differential privacy parameter',
      default=1e-3,
      type=float)
  parser.add_argument(
      '--teacher_no',
      help='the number of teachers in PATE framework (should be optimized)',
      default=100,
      type=int)
  parser.add_argument(
      '--lamda',
      help='hyper-parameter of DPBag on DP noise (should be optimized)',
      default=0.02,
      type=int)
  parser.add_argument(
      '--part_no',
      help='the number of partitions (should be optimized)',
      default=50,
      type=int)
  parser.add_argument(
      '--exp_iteration',
      help='the number of experiment iterations',
      default=5,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  metrics = main(args)