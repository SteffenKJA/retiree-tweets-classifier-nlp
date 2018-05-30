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
#from rpy2.robjects.packages import importr
import rpy2.robjects as ro
#import pandas.rpy.common as com
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()

ro.r('x=c()')
ro.r('x[1]=22')
ro.r('x[2]=44')
print(ro.r('x'))
print(ro.r['x'])

print(type(ro.r('x')))

ro.r('data(mtcars)')
#pydf = com.load_data('mtcars')
#print(pydf)

print("Done!")

#-----------------------------------------------------------------#
# March 4, 2015
# Cliff Voetelink
# Building a Tweet based classification model       
#-----------------------------------------------------------------#

# Instructions
# 1. Follow instructions in readRetiredFilesFinal.R
# 2. Follow instructions in readNormalFilesFinal.R
# 3. Run this script

#-------------------Install & Load Packages-----------------------#

#source("http://bioconductor.org/biocLite.R")
#biocLite("Rgraphviz")

ro.r('vPackages <- c("stringr","plyr","dplyr","sampling","ggplot2","wordcloud","RWeka","Rgraphviz","qdapRegex")')

#install.packages("rJava",type='source', repos='http://cran.us.r-project.org')
#install.packages("Rgraphviz", repos='http://cran.us.r-project.org')
#library("Rgraphviz")
#install.packages("qdapRegex", repos='http://cran.us.r-project.org')
#library("qdapRegex")

#lapply(vPackages, install.packages, character.only=T)
ro.r('lapply(vPackages, require, character.only=T)')
#install.packages("stringr", repos='http://cran.us.r-project.org')
#library("stringr")
#install.packages("plyr", repos='http://cran.us.r-project.org')
#library("plyr")
#install.packages("dplyr", repos='http://cran.us.r-project.org')
#library("dplyr")
#install.packages("sampling", repos='http://cran.us.r-project.org')
#library("sampling")
#install.packages("ggplot2", repos='http://cran.us.r-project.org')
#library("ggplot2")
#install.packages("wordcloud", repos='http://cran.us.r-project.org')
#library("wordcloud")
#install.packages("RColorBrewer", repos='http://cran.us.r-project.org')
#library("RColorBrewer")
#install.packages("RWeka", repos='http://cran.us.r-project.org')
#library("RWeka")
#install.packages("Rgraphviz", repos='http://cran.us.r-project.org')
#library("Rgraphviz")
#install.packages("qdapRegex", repos='http://cran.us.r-project.org')
#library("qdapRegex")

# ------------- run readRetiredFilesFinal.R instructions -----------#
ro.r('setwd("PAConsulting/tweets-hash")')
ro.r('source("readRetiredFilesFinal.r")')
#source("cleanFunctions.r")
#source("getFunctions.r")
#require("stringr")

ro.r('filePath                <- "./retired-hash/"')
ro.r('vFileNamesRetired       <- list.files(filePath)')
ro.r('lMyDataRetired          <- getAllDataListRetired(filePath, vFileNamesRetired)')

#tes = r.data('lMyDataRetired')
#print(type(tes))
#print(tes)

#df = pandas2ri.ri2py(r.data('lMyDataRetired'))
#print(df)
#print("Done with retired")

print("Working...")

# ------------- run readNormalFilesFinal.R instructions -----------#
ro.r('source("readNormalFilesFinal.R")')
#setwd("./PAConsulting/tweets-hash")
#source("cleanFunctions.R")
#source("getFunctions.R")
#require("stringr")

ro.r('filePath         <- "./normal-hash/"')
ro.r('vFileNamesNormal <- list.files(filePath)')

ro.r('lMyDataNormal    <- list()')
ro.r('lMyDataNormal    <- getAllDataListNormal(filePath, vFileNamesNormal)')
ro.r('lMyDataNormalAdj <- joinDfSameMonths(lMyDataNormal)')

# ---------------------------Set-Up--------------------------------#


#setwd("./PAConsulting/tweets-hash")

ro.r('lMyData        <- append(lMyDataNormalAdj, lMyDataRetired)')
ro.r('dfMyDataRaw    <- rbind.fill(lMyData)')

ro.r('dfMyDataRaw$vTweets  <- sapply(dfMyDataRaw$vTweet, toString)')

ro.r('dfMyData <- dfMyDataRaw')

# Extract datasets
r.data('dfMyData')
dfMyData = pandas2ri.ri2pandas(r['dfMyData'])

os.chdir('/Users/Steffen_KJ/Dropbox/Nets')
dfMyData.to_csv('dfMyData.csv')

#r.data('dfMyDataRaw$vTweets')
#dfMyDatavTweets = pandas2ri.ri2pandas(r['dfMyDataRaw$vTweets'])
#dfMyDatavTweets.to_csv('dfMyDatavTweets.csv')
#print(dfMyDatavTweets)
# Write dataframes to csv file for ease of use

#ro.r("write.csv(dfMyData, 'test.csv')")
#df_load = pd.read_csv(file, na_values=['NA'])
#print(df_load)

#print(r.data['dfMyData'])

#df = pandas2ri.ri2py(r.data('dfMyData'))
#print(df)
print("Done")
