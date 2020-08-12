# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:33:51 2019

@author: Administrator
"""

import numpy as np
import scipy.io as sio
from collections import Counter
from sklearn.utils import class_weight
import math
import sys
import tensorflow as tf


#%%produce imbalanced data
def imbalanced_data(final_data, final_label, imbalanced_dict, refresh=False, seed=1, name='imb_dataset'):
    np.random.seed(seed)
#    path = r'F:\Datasets\DL\Compound_dataset\\'+name+'_'+str(seed)+'.h5'
#    try:
#        if refresh:
#            train_X, train_Y, test_X, test_Y = ddio.io(path)
#        else:
#            train_X, train_Y, test_X, test_Y = ddio.load(path)
#    except:
    final_label_1=[np.argmax(one_hot)for one_hot in final_label]
    final_label_2=np.array(final_label_1)
    sample_num=Counter(final_label_2)[0]
    imbalanced_dict = {key: np.ceil(value/max(imbalanced_dict.values())*sample_num) 
                       for key, value in imbalanced_dict.items()}
    
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    #print(final_label)
    for each_label in range(int(np.max(final_label_2))+1):
        ratio = imbalanced_dict[each_label]
        index = np.where(final_label_2 == each_label)[0]
        #print(index)
        sample_index = np.random.choice(index, int(ratio), replace=False)
        #print(sample_index)
        parameters = int(np.ceil(0.6*np.size(sample_index)))
        #print(parameters)
    #    train_1=x_data[sample_index[:parameters],:]
    #    train_1=np.array(train_1)
    #    train_X.append(train_1)
    #    train_1=np.array(train_1)            
        train_X.append(final_data[sample_index[:parameters],:])
        train_Y.append(final_label_2[sample_index[:parameters]])
        #print(train_Y)
        test_X.append(final_data[sample_index[parameters:],:])
        test_Y.append(final_label_2[sample_index[parameters:]])
    #
    #train_X=np.array(train_X)
    #train_Y=np.array(train_Y)
    train_X, train_Y = np.concatenate(train_X), np.concatenate(train_Y)
    test_X, test_Y = np.concatenate(test_X), np.concatenate(test_Y)
    #train_X, train_Y = shuffle_data(train_X, train_Y)
    #test_X, test_Y = shuffle_data(test_X, test_Y)
    
#    ddio.save(path, [train_X, train_Y, test_X, test_Y])

    return train_X, train_Y, test_X, test_Y,imbalanced_dict
    
#%%shuffle data
def shuffle_data(no_shuffle_data, no_shuffle_labels):
    np.random.seed(1)
    total_num = np.shape(no_shuffle_data)[0]
    total_index = np.random.permutation(total_num)
    return no_shuffle_data[total_index, :], no_shuffle_labels[total_index]


#%%calculate class_weight
def calculate_class_weigh(y):
    y_integers = np.argmax(y, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
    d_class_weights = dict(enumerate(class_weights))
    return d_class_weights
    
    
#%%design class_weight
def create_class_weight(y):
    #total=np.sum(y.values())
    keys=y.keys()
    class_weight_1=dict()
    for key in keys:
        score=(max(y.values()))/(y[key])
        class_weight_1[key]=score
    
    class_weight_1[0] = class_weight_1[0]/2
    class_weight_1[4] = class_weight_1[4]*3
    class_weight_1[5] = class_weight_1[5]*2
    class_weight_1[6] = class_weight_1[6]*2
    
    return class_weight_1
    
#%%sample weight
def creat_sample_weight(y,weight,split):
    keys=y.keys()
    sample_weight=[]
    class_weight=[]
    #a=[[]]
    for key in keys:
        a=int(y[key])
        a=math.ceil(a*split)
        b=weight[key]
        class_weight.append(b)
        c=[b]*a
        sample_weight.append(c)
        #a=np.tile(weight[key],(y[key],1))
        #sample_weight.append(a)
    d=np.concatenate((sample_weight[0],sample_weight[1]),axis=None)
    d=np.concatenate((d,sample_weight[2]),axis=None)
    d=np.concatenate((d,sample_weight[3]),axis=None)
    d=np.concatenate((d,sample_weight[4]),axis=None)
    d=np.concatenate((d,sample_weight[5]),axis=None)
    d=np.concatenate((d,sample_weight[6]),axis=None)
    d=np.concatenate((d,sample_weight[7]),axis=None)
    return d,class_weight
#%%训练数据的mini_batch
def batches(batch_size, features, labels, weights):
    assert len(features) == len(labels) == len(weights)
    
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i], weights[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches
    
#%%训练数据的mini_batch
def cnn_batches(batch_size, features, labels):
    assert len(features) == len(labels)
    
    output_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches
    
#%%guiyihua
def norm_ZS(data):
    data_norm=[]
    for i in range(np.shape(data)[1]):
        data_max=data[:,i].mean()
        data_min=math.sqrt(data[:,i].var())
        x=(data[:,i]-data_max)/data_min
        data_norm.append(x)
    data_norm=np.array(data_norm)
    data_norm=data_norm.T
    return data_norm
    
#%%jindutiao
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


class ProgressBar():

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
        self.progress_width = 50

    def update(self, step=None):
        self.current_step = step

        num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
        num_rest = self.progress_width - num_pass 
        percent = (self.current_step+1) * 100.0 / self.max_steps 
        progress_bar = '[' + '■' * (num_pass-1) + '▶' + '-' * num_rest + ']'
        progress_bar += '%.2f' % percent + '%' 
        if self.current_step < self.max_steps - 1:
            progress_bar += '\r' 
        else:
            progress_bar += '\n' 
        sys.stdout.write(progress_bar) 
        sys.stdout.flush()
        if self.current_step >= self.max_steps:
            self.current_step = 0
            print

#%%g-mean,f-score
def count_nums(true_labels, num_classes):
    initial_value = 0
    list_length = num_classes
    list_data = [ initial_value for i in range(list_length)]
#    for i in range(0, num_classes):
#        list_data[i] = true_labels.count(i)
    list_data=tf.bincount(true_labels,dtype=tf.int32)
    return list_data   

def accuracy(confusion_matrix, true_labels, num_classes):
    list_data = count_nums(true_labels, num_classes)
 
    initial_value = 0
    list_length = num_classes
    true_pred = [ initial_value for i in range(list_length)]
    for i in range(0,num_classes-1):
        true_pred[i] = confusion_matrix[i][i]

    acc = []
    for i in range(0,num_classes-1):
        acc.append(0)
 
    for i in range(0,num_classes-1):
        acc[i] = true_pred[i] / list_data[i]
 
    return acc

def accr_confusion_multiply(accr_confusion,num_classes):
    accr_confusion_multiply=tf.constant(1.0,dtype=tf.float64)
    for i in range(0,num_classes-1):
        accr_confusion_multiply=accr_confusion_multiply*accr_confusion[i]
    return accr_confusion_multiply
   