#-----------------------------------------------#
# Febr 22, 2014
# Cliff Voetelink
# The script below reads all the data from normal users,
# stores the relevant parts in data frames (1 per file).
# Then it returns a list of these data frames.
# It finally returns a list of data frames where 
# each data frame corresponds to the data of 1 month.
#-----------------------------------------------#

#Instructions:
#1. Load all the functions below from next section onwards
#2. Run the following 10 lines of code in this section. 

#Note: By default, due to memory capacity, we only read in 5000 lines per file.

#rm(list = ls(all = TRUE)) 
setwd("./PAConsulting/tweets-hash")
source("cleanFunctions.R")
source("getFunctions.R")
require("stringr")

filePath         <- "./normal-hash/"
vFileNamesNormal <- list.files(filePath) 

lMyDataNormal    <- list()
lMyDataNormal    <- getAllDataListNormal(filePath, vFileNamesNormal)
lMyDataNormalAdj <- joinDfSameMonths(lMyDataNormal)

#-----------------------------------------------#


getAllDataListNormal <- function(filePath, vFileNames){
        
        # Reads all files, resulting data frame of each file is put into list 
        # names of list are names of files minus .xls.csv
        
        vFileRef        <- getFullFilePath(filePath,vFileNames)
        lMyData         <- lapply(vFileRef, getDataDfNormal)        
        names(lMyData)  <- getFileTitle(vFileRef)
        
        return(lMyData)
        
}


getDataDfNormal <- function (fileRef){
       
        # 1. Reads all data from the file
        # 2. Gets all the relevant rows and returns intermediate results as data frame vRelevantLinePart, pattern_id, sender_id, vRawTweets]
        # 3. Processes vRawTweets 
        # 4. Finally puts 3 hashkeys along with preprocessed tweets into a dataframe. Determines which rows have relevant [valid] tweets.
        # 5. Output: Return the subset of this dataframe with valid tweets
        
        cat("Currently reading the following file: ", fileRef, "\n")
        
        # Pattern 1: 3 Hashkeys, Tweet, Date 
        
        sPattern1 <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),(.{1,160}),(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4})"
       
        # Pattern 2: 3 Hashkeys, Date, Tweet, Score, User Description
        
        sPattern2 <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4},(.{1,160}),([0-9]{2}),([a-zA-Z0-9]{50,})"
        vPattern  <- c(sPattern1, sPattern2)     

        vLines                  <- readXlsCsvFile(fileRef)
        dfRelevantRows          <- getdfRelevantRows(vLines, vPattern)       
                        
        vTweets                 <- preProcessRawTweetsNormal(dfRelevantRows$vRawTweets)     
        dfMyData                <- createDataDf(dfRelevantRows$vRelevantLinePart, vTweets, fileRef, "Normal")        
        vIndexRelevant          <- getRelevantTweetIndices(vTweets)
        
        return(dfMyData[vIndexRelevant,])
        
}


readXlsCsvFile <- function(fileRef){
        
        #Reads the lines in the Xls.CSV file 
        #Returns a vector of Strings of each line in the input
        
        conn     <- file(fileRef,open="r")        
        vLines   <- readLines(conn, n=5000) #remove, n=5000 for whole file   
        
        close(conn)        
        
        return(vLines)
} 

getdfRelevantRows <- function(vLines, vPattern){
                
        # Gets and saves rows where input satisfies a pattern in vPattern
        # Output: returns vRelevantLinePart, pattern_id, sender_id, vRawTweets as a data frame
        # where vRelevantLinePart is either the full line or the full line with a cutoff after the tweet
   
        dfRelevantRows          <- as.data.frame(t(sapply(vLines,getDfRow, vPattern)))
        dfRelevantRows          <- dfRelevantRows[!is.na(dfRelevantRows[,1]),]        
        names(dfRelevantRows)   <- c("vRelevantLinePart", "pattern_id", "sender_id", "vRawTweets") 
        
        return(dfRelevantRows) 
        
}


getDfRow <- function(sLine, vPattern){
        
        #Returns a data frame row: vRelevantLinePart, pattern_id, sender_user_id, raw tweet
        #Where vRelevantLinePart denotes a partial full line including the tweet
        
        sLinePat1 <- str_extract(sLine, vPattern[1])                
        
        if(!is.na(sLinePat1)){
                
                keyValuePair    <- getSenderAndRawTweet(sLinePat1, 1)                                      
                dfRow           <- cbind(sLinePat1, 1, keyValuePair[1], keyValuePair[2])

                
        } else {
                
                sLinePat2 <- str_extract(sLine, vPattern[2]) 
                
                if(!is.na(sLinePat2)){
                        
                        keyValuePair    <- getSenderAndRawTweet(sLinePat2, 2)
                        dfRow           <- cbind(sLinePat2, 2, keyValuePair[1], keyValuePair[2])
                        
                } else {
                        
                        dfRow            <- cbind(NA,NA,NA,NA)
                }
                
                
        }
        
        return(dfRow)
        
        
}



getSenderAndRawTweet <- function (sLine, pattern_id){
        
        # Input: line + and a pattern_id (1 or 2) 
        # Output: returns vector of Strings, containing sender_id and rawTweet        
        # sender_id = hashed version of interaction.author.username
        # rawTweet
        
        #Pattern 1A: 3 Hashkeys, User Name, Date Time, Date or Hashkey  
        sPattern1A    <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9_.]){1,15}(\\s)?(([a-zA-Z0-9_.]){1,15})?,(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4},((\\s)?(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4})|([0-9]{2},[0-9a-zA-Z]{50}))"
        sLinePat1A    <- str_extract(sLine, sPattern1A)  
        
        #Pattern 1B: 3 Hashkeys, User Name, Date 
        sPattern1B    <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9_.]){1,15}(\\s)?(([a-zA-Z0-9_.]){1,15})?,(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4})"
        sLinePat1B    <- str_extract(sLine, sPattern1B) 
        
        #Pattern 1C:3 Hashkeys, Date Time, City /  Country, Date | Number
        sPattern1C   <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),((Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4},)?([A-Z][A-Za-z]{1,15})(\\s)?([A-Z][A-Za-z]{1,15})?,(\\s)?([A-Z][A-Za-z]{1,15})?(\\s)?([A-Z][A-Za-z]{1,15})?,(((Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}))|([0-9]{2}))"
        sLinePat1C   <- str_extract(sLine, sPattern1C)  
        
        #Pattern 2A: 3 Hashkeys, Date Time, Username, Date or Hashkey
        sPattern2A    <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4},([a-zA-Z0-9_.]){1,15}(\\s)?(([a-zA-Z0-9_.]){1,15})?,((\\s)?(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4})|([0-9]{2},[0-9a-zA-Z]{50}))"
        sLinePat2A    <- str_extract(sLine, sPattern2A)  
        
        #Pattern 2B: 3 Hashkeys, Date Time, Date
        sPattern2B    <- "([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4},(\\s)?(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4})"
        sLinePat2B    <- str_extract(sLine, sPattern2B)          
        
        if(pattern_id == 1 & is.na(sLinePat1A) & is.na(sLinePat1B) & is.na(sLinePat2A) & is.na(sLinePat2B) & is.na(sLinePat1C)){
                
                vCommaPositions <- getAllCommaPositions(sLine)     
                sRawTweet       <- getRawTweet(sLine, pattern_id, vCommaPositions)
                sSenderID       <- getSenderID(sLine, vCommaPositions)
                vResult         <- c(sSenderID,sRawTweet)                 
               
                return(vResult)
                
        } else if (pattern_id == 2 & is.na(sLinePat2A) & is.na(sLinePat2B) & is.na(sLinePat1C)){
                
                
                vCommaPositions <- getAllCommaPositions(sLine)     
                sRawTweet       <- getRawTweet(sLine, pattern_id, vCommaPositions)
                sSenderID       <- getSenderID(sLine, vCommaPositions)
                vResult         <- c(sSenderID,sRawTweet) 
                
                return(vResult)
                
        } else {                        
                
                return(c(NA,NA))                       
        }
        
        return(c(NA,NA))
}


getRawTweet <- function(sLine, pattern_id, vCommaPositions = NA){
        
        # Given read-in row as a string, the pattern id and the comma positions of the line
        # Returns the rawTweet as a String
        
        if(length(vCommaPositions) == 1 & is.na(vCommaPositions[1])){
                
                vCommaPositions <- getAllCommaPositions(sLine)
        }
        
        if(pattern_id == 1 ){
                
                beginTweetIndex <- vCommaPositions[3] + 1
                endTweetIndex   <- vCommaPositions[length(vCommaPositions)-1] - 1                
                sRawTweet       <- substr(sLine, beginTweetIndex, endTweetIndex)
                
        } else if (pattern_id == 2){
                
                beginTweetIndex = vCommaPositions[5] + 1
                endTweetIndex   = vCommaPositions[length(vCommaPositions)-1] - 1
                sRawTweet       = substr(sLine, beginTweetIndex, endTweetIndex)
                
        } else {  # not possible
                
                sRawTweet <- NA
        }
        
        return(sRawTweet)              
        
}


getSenderID <- function (sLine, vCommaPositions = NA){        
        
        # Returns the hashed version of interaction.author.username
        
        if(length(vCommaPositions) == 1 & is.na(vCommaPositions[1])){
                
                vCommaPositions <- getAllCommaPositions(sLine)
        }
        
        beginSenderIndex = vCommaPositions[2] + 1
        endSenderIndex   = vCommaPositions[3] - 1        
        
        sSenderID         <- substr(sLine, beginSenderIndex, endSenderIndex)
        
        return(sSenderID)               
        
}



getAllCommaPositions <- function (x){
        
        #Gets all commma positions of a string x
        #Returns the comma positions in a vector
        
        vCommaPositions <- str_locate_all(x, ',')[[1]][,1]         
        return(vCommaPositions)
        
}


createDataDf <- function(vFullLines, vTweets, sFileRef, sClassification){   
        
        #Returns data frame with following columns: 
        #gender, sender_name, sender_id, fileTitle, target (Retired/Normal), vTweet
        
        df        <- getFirstThreeHashKeys(vFullLines)
        fileTitle <- as.factor(getFileTitle(sFileRef))
        target    <- as.factor(sClassification)
        df        <- cbind(df, fileTitle, target, vTweets)  
        
        return(df)
}



preProcessRawTweetsNormal <- function(vStrings) {
        
        #Input: Raw Tweets
        #Output: String vector of cleaner tweets
        
        vResult <- removeHashAndBeyond(vStrings)
        vResult <- removeDatesTimes(vResult,"Normal")
        vResult <- subCharBySpace(vResult,"\t")
        vResult <- subCharBySpace(vResult,'\"')       
        vResult <- str_trim(vResult, side = "both")
        vResult <- gsub("\\s+"," ", vResult)  
        vResult <- gsub("(\\s)?&amp(;)?(\\s)"," and ", vResult)
        
        return(vResult)
        
}

joinDfSameMonths <- function(lData){
        
        #Input:  List of Data Frames
        #Output: List of Data Frames where data frames belonging to the same month are united into 1 df
        #Assumes names of list elements are sorted in Month_i_Part1 Month_i_Part2 order
   
        lDataNew  <- list()
        vNamesNew <- vector()
                
        vNames   <- gsub("Part[0-9]", "", names(lData))            
        k <- 0
        
        if(length(vNames) == 1){
                
                k <- k + 1
                lDataNew[[k]]           <- lData[[1]]
                names(lDataNew[k])      <- vNames[1]
                
                return(lDataNew)
                
        }
        
        for (i in 1:(length(vNames)-1)){
                
                if(identical(vNames[i],vNames[i+1]) == TRUE){
                        
                        k <- k + 1
                        
                        lData[[i]]$fileTitle    <- gsub('Part[0-9]',"",lData[[i]]$fileTitle)
                        lData[[i+1]]$fileTitle  <- gsub('Part[0-9]',"",lData[[i+1]]$fileTitle)
                        
                        lDataNew[[k]]           <- rbind(lData[[i]],lData[[i+1]])
                        vNamesNew[k]            <- vNames[i]
                        
                } else if (i == 1){
                        
                        #No match but still add first df
                        
                        k <- k + 1
                        lDataNew[[k]]           <- lData[[i]]
                        vNamesNew               <- vNames[i]
                        
                } else if (i == (length(vNames)-1)){   
                        
                        #Last one is not a match but still add last df 
                        
                        k <- k + 1
                        lDataNew[[k]]           <- lData[[i+1]]         
                        vNamesNew               <- vNames[i+1]
                }

        }
        
        names(lDataNew) <- vNamesNew

        return(lDataNew)

}
