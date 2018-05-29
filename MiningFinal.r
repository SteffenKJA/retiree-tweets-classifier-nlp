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

vPackages <- c("stringr","plyr","dplyr","sampling","ggplot2","wordcloud","RWeka","Rgraphviz","qdapRegex")

#lapply(vPackages, install.packages, character.only=T)
lapply(vPackages, require, character.only=T)

#---------------------------Set-Up--------------------------------#


setwd("./PAConsulting/tweets-hash")

lMyData        <- append(lMyDataNormalAdj, lMyDataRetired) 
dfMyDataRaw    <- rbind.fill(lMyData)

dfMyDataRaw$vTweets  <- sapply(dfMyDataRaw$vTweet, toString)

#Working Dataset
dfMyData <- dfMyDataRaw 

