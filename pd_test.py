#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:50:59 2018

@author: Steffen_KJ
"""
import numpy as np
import pandas as pd

target = pd.Series([True, True, False, False])

res = pd.Series([True, True, False, True])

comb = target + res
target_arr = pd.Series(range(4))

res_arr = target_arr[comb]
print(res_arr)

print(res_arr)

print('len', len(res_arr))
