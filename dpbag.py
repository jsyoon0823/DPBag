"""DPBag Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate", 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate

Last updated Date: April 26th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

dpbag.py

(1) Use train and valid datasets to make differentially private classification model
(2) Use test set to measure the performances on different epsilons

Note: we use logistic regression as student and teacher models (as an example)
"""

# Necessary Packages
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression


def dpbag (train_x, train_y, valid_x, test_x, test_y, parameters):
  """DPBag module.
  
  Args:    
    - train_x: training features
    - train_y: training labels
    - valid_x: validation features
    - test_x: testing features
    - test_y: testing labels
    - parameters: differential privacy and DPBag parameters
      - Differential privacy parameters
        - epsilon
        - delta    
      - DPBat parameters (should be optimized for different datasets)
        - teacher_no: the number of teachers in PATE framework
        - lamda: hyper-parameter of DPBag on DP noise
        - part_no: the number of partitions    
        
  Returns:    
    - performance on testing data
      - out_acc: accuracy
      - out_auc: area under roc curve
      - out_apr: average precision recall
      - out_budget: privacy budget (the number of samples to be seen by DPBag)
  """  
  # Parameters
  no, dim = train_x.shape
                    
  epsilon    = parameters['epsilon']
  delta      = parameters['delta'] 
  lamda      = parameters['lamda'] 
  teacher_no = parameters['teacher_no'] 
  part_no    = parameters['part_no']
  
  # Initialize alpha
  L = 80
  alpha = [[0 for i in range(no)] for j in range(L)]
        
  ## Partition the training data (Divide data into multiple partitions)
  # Initialize partitions
  part_x = list()
  part_y = list()
    
  # Save the partition number and teacher number for each sample
  part_save = np.zeros([no, part_no, teacher_no])
    
  # For each partition
  for p in range(part_no):
    data_x = list()
    data_y = list()
    # Divide them into multiple disjoint sets (# of teachers)
    idx = np.random.permutation(no)
    # For each teacher    
    for i in range(teacher_no):
      # Index of samples in each disjoint set
      start_idx = i * int(no/teacher_no)
      end_idx = (i+1) * int(no/teacher_no)
      temp_idx = idx[start_idx:end_idx]            
      # Divide the data
      data_x.append(train_x[temp_idx,:])
      data_y.append(train_y[temp_idx])            
      # Save the teacher number and partition number
      part_save[temp_idx,p,i] = 1
        
    # Save each partition
    part_x.append(data_x)
    part_y.append(data_y)
        
  print('Finish data division')  
  ## Train teacher models
  # Initialize teacher models
  teacher_models = list()
    
  # For each partition
  for p_idx in (range(part_no)):        
    # Initialize teacher models for each partition
    part_models = list()      
    # For each teacher
    for t_idx in (range(teacher_no)):              
      # Load data for each teacher in each partition
      x_temp = part_x[p_idx][t_idx]
      y_temp = part_y[p_idx][t_idx]            
      # Train the teacher model
      model = LogisticRegression(solver='lbfgs')
      model.fit(x_temp, y_temp)         
      # Save the teacher model
      part_models.append(model)                
    # Save the teacher model in each partition
    teacher_models.append(part_models)
    
  print('Finish teacher training')
            
  ## Train a student model
  # Initialize some parameters
  # Track current epsilon
  epsilon_hat = 0
  # Track the privacy budget
  mb_idx = 0
  # Set the public data
  no, _ = valid_x.shape
  x_mb = valid_x[:no,:]
  # r_s initialize
  r_mb = np.zeros([no, 1])
  
  # Output Initialization (Accuracy, AUC, APR, Privacy Budget)
  out_acc = list()
  out_auc = list()
  out_apr = list()
  out_budget = list()
  
  ## Get all the n_c, n_c(x), and m(x) to speed up the algorithm
  # Tx_all (T_i,j(x)) Initialization
  Tx_all = np.zeros([part_no, teacher_no, no])
               
  # Outputs of all teachers for public data
  for p_idx in (range(part_no)):
    for t_idx in range(teacher_no):
      teacher_pred_result_temp = teacher_models[p_idx][t_idx].predict_proba(x_mb)[:,1]          
      # Save them to the T_i,j(x)
      Tx_all[p_idx, t_idx, :] = np.reshape(1*(teacher_pred_result_temp>0.5), [-1])
    
  ## Compute nc_all (n_c)
  nc_all = np.zeros([no,2])    
  nc_all[:,0] = np.sum(1-Tx_all, axis = (0,1)) / part_no
  nc_all[:,1] = np.sum(Tx_all, axis = (0,1)) / part_no
    
  # Compute ncx_all (n_c(x))
  ncx_all = np.zeros([no, no, 2])    
      
  ncx_all[:,:,1] = np.einsum('npt,ptc -> nc', part_save, Tx_all) / part_no
  ncx_all[:,:,0] = 1 - ncx_all[:,:,1]
    
  # Compute m(x) 
  mx_all = np.max(ncx_all, axis = 2)  
    
  print('Finish nc, ncx, mx computation')

  ## Get access to the data until the epsilon is less than the threshold & budget is less than the public data
  while ((epsilon_hat < epsilon) & (mb_idx < no)): 
    # PATE_lambda (x)
    r_mb[mb_idx,0] = np.argmax([nc_all[mb_idx,0] + np.random.laplace(scale=1/lamda), nc_all[mb_idx,1] + np.random.laplace(scale=1/lamda)])  
    
    # Compute alpha    
    for l_idx in range(L):
      first_term  = (2 * ( (lamda*mx_all[:,mb_idx])**2 ) * (l_idx + 1) * (l_idx + 2)) 
      alpha[l_idx] = alpha[l_idx] + first_term
      
    # compute epsilon hat       
    min_list = list()        
    for l_idx in range(L):
      temp_min_list = (np.max(alpha[l_idx]) + np.log(1/delta)) / (l_idx+1)        
      min_list.append(temp_min_list)
                 
    ## For each int epsilon boundary
    if (int(epsilon_hat) < int(np.min(min_list))):
        
      # Student Training
      # Use entire data until int(epsilon_hat) < int(np.min(min_list))
      s_x_train = x_mb[:mb_idx,:]
      s_y_train = r_mb[:mb_idx,:]
                            
      # Train the DP classification model
      model = LogisticRegression(solver='lbfgs')
      model.fit(s_x_train, s_y_train)
            
      # Evaluations
      student_y_final = model.predict_proba(test_x)[:,1]
      student_pred_result = roc_auc_score (test_y, student_y_final)  
            
      print('Student AUC: ' +str(np.round(student_pred_result,4)) + ', Epsilon: ' + str(np.round(epsilon_hat)))
            
      out_acc.append(np.round(accuracy_score(test_y, student_y_final > 0.5),4))
      out_auc.append(np.round(roc_auc_score (test_y, student_y_final),4))
      out_apr.append(np.round(average_precision_score (test_y, student_y_final),4))
      out_budget.append(mb_idx+1)
            
    ## Epsilon, mb_update Update
    epsilon_hat = np.min(min_list)
    # The number of accessed samples
    mb_idx = mb_idx + 1
    # Print current state (per 1000 accessed samples)
    if (mb_idx % 1000 == 0):            
      print('step: ' + str(mb_idx) + ', epsilon hat: ' + str(epsilon_hat))        
  
  # Return Accuracy, AUC, APR and Privacy Budget
  return out_acc, out_auc, out_apr, out_budget