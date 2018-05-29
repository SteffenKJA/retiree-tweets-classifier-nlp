#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:07:55 2018

@author: Steffen_KJ
"""

# This is a script solving the Nets data scientist case.

import numpy as np
import scipy as sp
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

ro.r('x=c()')
ro.r('x[1]=22')
ro.r('x[2]=44')
print(ro.r('x'))
print(ro.r['x'])

print(type(ro.r('x')))

ro.r('data(mtcars)')

pydf = com.load_data('mtcars')

