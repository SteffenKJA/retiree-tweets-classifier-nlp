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

#install.packages("rJava",type='source', repos='http://cran.us.r-project.org')
#install.packages("Rgraphviz", repos='http://cran.us.r-project.org')
#library("Rgraphviz")
#install.packages("qdapRegex", repos='http://cran.us.r-project.org')
#library("qdapRegex")

#lapply(vPackages, install.packages, character.only=T)
lapply(vPackages, require, character.only=T)

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
setwd("PAConsulting/tweets-hash")
source("readRetiredFilesFinal.r")
#source("cleanFunctions.r")
#source("getFunctions.r")
#require("stringr")

filePath                <- "./retired-hash/"
vFileNamesRetired       <- list.files(filePath)
lMyDataRetired          <- getAllDataListRetired(filePath, vFileNamesRetired)


# ------------- run readNormalFilesFinal.R instructions -----------#
source("readNormalFilesFinal.R")
#setwd("./PAConsulting/tweets-hash")
#source("cleanFunctions.R")
#source("getFunctions.R")
#require("stringr")

filePath         <- "./normal-hash/"
vFileNamesNormal <- list.files(filePath)

lMyDataNormal    <- list()
lMyDataNormal    <- getAllDataListNormal(filePath, vFileNamesNormal)
lMyDataNormalAdj <- joinDfSameMonths(lMyDataNormal)

#---------------------------Set-Up--------------------------------#


#setwd("./PAConsulting/tweets-hash")

lMyData        <- append(lMyDataNormalAdj, lMyDataRetired) 
dfMyDataRaw    <- rbind.fill(lMyData)

dfMyDataRaw$vTweets  <- sapply(dfMyDataRaw$vTweet, toString)

#Working Dataset
dfMyData <- dfMyDataRaw

print("Done")

