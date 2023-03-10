# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:35:03 2022

@author: 柴文清
"""

from sklearn.metrics import accuracy_score
import numpy as np


from math import log

from math import log
def entropy(Cluster):
    N_c = sum(Cluster)
    e = 0
    for Point in Cluster:
        if Point > 0:
            e += (Point / N_c) * log(Point / N_c,2)
    return -e
def entropy_total(Clusters):
    m = 0
    e = 0
    e_total = 0
    for Cluster in Clusters:
        N_c = sum(Cluster)
        m += N_c
        e += entropy(Cluster) * (N_c)
    e_total = e / m
    return e_total
 
  
 
def purity(Clusters):
    m = 0
    p = 0
    for Cluster in Clusters:
        m += sum(Cluster)
        p += max(Cluster)
    P = p / m        
    return P     

