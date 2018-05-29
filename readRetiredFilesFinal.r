#-----------------------------------------------#
# Febr 22, 2014
# Cliff Voetelink
# The script below reads all the data from retired users
# and returns the relevant parts as a list of data frames
# The data of each file will be in a separate data frame
#-----------------------------------------------#

#Instructions:
#1. Load all the functions below from next section onwards
#2. Run the following 7 lines of code from this section

#rm(list = ls(all = TRUE)) 
setwd("./PAConsulting/tweets-hash")
source("cleanFunctions.R")
source("getFunctions.R")
require("stringr")

filePath                <- "./retired-hash/"
vFileNamesRetired       <- list.files(filePath) 
lMyDataRetired          <- getAllDataListRetired(filePath, vFileNamesRetired)

#-----------------------------------------------#


getAllDataListRetired <- function(filePath, vFileNames){
        
        # Reads all txt files, resulting data frame of each file is put into list 
        # names of list are names of files minus .txt
        # Returns list of dataframes where each df corresponds to a txt file
        
        vFileRef        <- getFullFilePath(filePath,vFileNames)
        lMyData         <- lapply(vFileRef, getDataDfRetired)        
        names(lMyData)  <- getFileTitle(vFileRef)
        
        return(lMyData)
        
}

getDataDfRetired <- function(fileRef){
        
        # 1. Reads all data from the file where line starts with 3 hashkeys
        # 2. Gets all tweets
        # 3. Puts data in data frame
        # 4. Finally produces 3 hashkeys along with relevant preprocessed tweets into a dataframe where there is a tweet (no retweet)
        # Returns the final dataframe
        
        cat("Currently reading the following file: ", fileRef, "\n")
        
        vFullLines      <- readTxtFile(fileRef)           
        vTweets         <- getAllTweets(vFullLines)     
        dfMyData        <- createDataDf(vFullLines, vTweets, fileRef, "Retired")
        vIndexRelevant  <- getRelevantTweetIndices(vTweets) 
        
        return(dfMyData[vIndexRelevant,])
}


createDataDf <- function(vFullLines, vTweets, sFileRef, sClassification){   
        
        #Returns data frame with following columns: 
        #gender, sender_name, sender_id, fileTitle, target (Retired/Normal), vTweets
        
        df        <- getFirstThreeHashKeys(vFullLines)
        fileTitle <- as.factor(getFileTitle(sFileRef))
        target    <- as.factor(sClassification)
        df        <- cbind(df, fileTitle, target, vTweets)  
        
        return(df)
}


readTxtFile <- function(fileRef){
        
        #Reads the txt file and returns a vector of Strings 
        #where the input follows the 3 hashkeys: Gender, author_name, author_user_id
                
        conn    <- file(fileRef,open="r")
        lines   <- readLines(conn) #remove n= 200 for whole file        
        
        sHashKeyPattern  <- '^(([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),(.*))'
        
        vLinesHashKey   <- str_extract(lines, sHashKeyPattern)          
        vHasHashIndex   <- !is.na(vLinesHashKey)
        
        close(conn)         
        return(vLinesHashKey[vHasHashIndex])
}



getAllTweets <- function (vFullLines){
                
        # Input = full lines that were read in        
        # Output = The tweets as string vector
        # Main Idea: find tweets that follow a specific patern, isolate tweets if there is one and put them in tweet vector
        # From that point on don't touch them and try to fill in the tweets that we don't have yet by going to next pattern
        
        vLines      <- removeTwitterDevice(removeFirst3HashKeys(vFullLines))     
        vTweets     <- rep(NA, length(vLines))        
        
        #Pattern 1: 3 Hashkeys, tweet\"description, \"\t1089\t2000\
        
        sPattern1   <- '(\\s)?.{1,150}\"\t.{1,200}\"\t[0-9]{1,10}\t[0-9]{1,10}'
        vTweets     <- getTweets(vLines, vTweets, sPattern1, 1, '\"\t')
        
        #Pattern 2: 3 Hashkeys, description, \"\t1089\t2000\ and no tweet
        
        sPattern2   <- '^(\\s)?.{1,200}\"\t[0-9]{1,10}\t[0-9]{1,10}'
        vLinePat2   <- str_extract(vLines, sPattern2)        
        vTweets     <- getTweets(vLines, vTweets, sPattern2, 2, '\"\t')
        
        #Pattern 3: 3 Hashkeys, description. Example: 3 hashkeys, " settled London. Journalist, very political animal,487b8ccd520d4ed31dbcbb806a9848594e89039c85e87e1b3b22cbc3,ff94c410fa425f5e0505d942c705557aeaf345e551037047337d0b10"                                                                                                                                                                                                                                                          
        
        sPattern3    <- '^(\\s)?.{1,160}\"(\t).{1,150},[A-Za-z0-9]{50,}'
        sPattern4    <- '^(\\s)?.{1,160},[A-Za-z0-9]{50,}'
        sPatternDate <- '^((\\s)[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4})'
        
        vLinePat3     <- str_extract(vLines,sPattern3)
        vLinePat4     <- str_extract(vLines,sPattern4)
        vLinePatDate  <- str_extract(vLines,sPatternDate)
        
        vBoolean        <- !is.na(vLinePat4) & is.na(vLinePat3) & is.na(vLinePatDate)      
        vRelevantIndex  <- which(vBoolean == TRUE)        
        vTweets         <- replaceNAtweets(vTweets, vector(), vRelevantIndex, "", 3, list())
                                                                                                                                                                                                                                                                                                                                                                                                                                        
        #Pattern 5: Strings that follow this type of pattern \t437\t272\tlondon\talex\t\t44986959\t{" 
        
        sPattern5       <- '^(\t[0-9]{1,10}\t[0-9]{1,10}(.*)\t[0-9]{1,10}(.*))'
        vLinePat5       <- str_extract(vLines,sPattern5)
        vRelevantIndex  <- which(!is.na(vLinePat5) == TRUE)  
        vTweets         <- replaceNAtweets(vTweets, vector(), vRelevantIndex, "", 5, list())
        
        #Remove Locations
        vLines <- removeLocationRetired(vLines)
        
        #Pattern 6: Date Followed by a lot of empty tabs
        sPattern6     <- '((Mon|Tue|Wed|Thu|Fri|Sat|Sun),)?(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4}\"(\t){1,20}\t[0-9]{1,2}(\t){1,20}\"'
        vLinePat6     <- str_extract(vLines, sPattern6)               
        
        vLines[is.na(vLinePat6)]  <- removeDoubleQuoteTabAndBeyond(vLines[is.na(vLinePat6)])  #If not matched with pattern 6
        vLines[!is.na(vLinePat6)] <- gsub(sPattern6,"",vLines[!is.na(vLinePat6)])             #If pattern 6 matched, remove it
        
        vTweets[is.na(vTweets)] <- vLines[is.na(vTweets)]
                        
        vResult <- preProcessRawTweetsRetired(vTweets)
        
        return(vResult)        
        
}


getTweets <- function(vStrings, vTweets, sPattern, pattern_id, specialChar){
        
        #For as specific pattern, it finds the locations of tweets using the positions of the specialChar in that pattern
        #It then replace the tweets for only those positions which are still denoted by NA for where we find a tweet
        #Returns vTweets
        
        if( pattern_id != 1 & pattern_id != 2){ 
                
                stop('Error: getTweets only works for pattern id 1 and 2. For other patterns, use replaceNAtweets.')
                
        }        
        
        vStringWithPattern <- str_extract(vStrings, sPattern) 
        
        vIndex          <- which(!is.na(vStringWithPattern) == TRUE)
        lPositions      <- getAllSpecialCharPositions(vStringWithPattern,specialChar)
        
        vTweets         <- replaceNAtweets(vTweets, vStringWithPattern, vIndex, sPattern, pattern_id, lPositions)
        
        return(vTweets)
}


replaceNAtweets <- function(vTweets, vStringWithPattern = vector(), vRelevantIndex, sPattern, pattern_id, lPositions = list()){       
        
        # For Tweets that still have value NA, we are going to replace them by the found tweet or "" given a pattern_id and positions of special characters
        # Returns vTweets
        
        if(pattern_id == 1){
                
                for (i in vRelevantIndex){
                        
                        if(is.na(vTweets[i]) == TRUE){
                                vTweets[i] <- substr(vStringWithPattern[i], 0, lPositions[[i]][1,1]-1)
                        }
                }
                
        }        
        
        if(pattern_id == 2 | pattern_id == 3 | pattern_id == 5){
                
                # In this case there is no tweet and we assign an empty string
                
                for (i in vRelevantIndex){
                        
                        if(is.na(vTweets[i]) == TRUE){
                                vTweets[i] <- ""
                        }
                }
                
        }
        
        return(vTweets)
        
        
}


preProcessRawTweetsRetired <- function(vStrings) {
        
        #Cleans Raw Data Lines and Returns cleaner tweets
        
        vResult <- removeHashAndBeyond(vStrings)
        vResult <- removeDatesTimes(vResult,"Retired")
        vResult <- removeUserInfo(vResult)
        vResult <- removeNames(vResult)    
        vResult <- removeNumberBetweenTabs(vResult) 
        vResult <- removeMultiTab(vResult)    
        vResult <- removeLocationRetired(vResult)
        vResult <- removeExceptions(vResult)        
        vResult <- removeGeoAndBeyond(vResult)
        vResult <- subCharBySpace(vResult,"\t")
        vResult <- subCharBySpace(vResult,'\"')
        vResult <- removeEndingInBracket(vResult)        
        vResult <- str_trim(vResult, side = "both")
        vResult <- gsub("\\s+"," ", vResult)  
        vResult <- gsub("(\\s)?&amp(;)?(\\s)"," and ", vResult)
        return(vResult)
        
}


removeMultiTab <- function(vStrings){
        
        #Removes Multiple Consecutive Tabs
        
        sPattern <- "[\t]{2,50}"
        
        return(gsub(sPattern,"",vStrings))
}


removeNames <- function(vStrings){
        
        #Removes User Names between 2 tabs incl. tabs itself
        
        sPattern <- "^(\t[A-Za-z0-9]{1,20}\t)"
        
        return(gsub(sPattern,"",vStrings))
}


removeNumberBetweenTabs <- function(vStrings){
        
        #Remove Numbers between 2 tabs
        
        sPattern <- "[\t]+[0-9]+[\t]+"
        
        return(gsub(sPattern,"",vStrings))
}


removeEndingInBracket <- function(vStrings){    
        
        #Replaces the whole line by "" if it follows the 
        #following pattern and ends in a bracket
        
        return(gsub("((.*)(\\s){2,30}\\])$","",vStrings))
        
}


removeUserInfo <- function(vString){
        
        # Removes User Data like the following
        # Example: \"\tMatt Paddock \t\t22480808\t\"{"                                                                                                                                                                                                                                                                                                                                                                                                                       
        # Example: "\tBarbara Green\t\t175471815\t\t\t\t" 
        
        sPattern <- '\t[a-zA-Z]{1,15}(\\s[a-zA-Z]{1,15})?(\\s[a-zA-Z]{1,15})?(\\.| )?\t\t[0-9]{1,15}(\t)+.*'
        
        return(sub(sPattern,"",vString))
}


removeExceptions <- function(vString){
        
        #Cleans the tweets that were not read in properly
        
        sPattern <- '(^[\\s]+[Uu][Kk]([\\.\\s]+)?)|(^([Uu].[Kk](\\.)?))|(^([Uu][Kk].))|(\"\\{)|(Essex, England)|(\\sHants\\.\\sUK)|(^(\\s)England\\.)|(,Yorkshire in the UK)|((.*)\tAndy \\| Mira(.*))|((.*)\"\"f1\"\"(.*))|"^((\\s)UK.)'
        
        return(sub(sPattern, "", vString))  
        
}

removeGeoAndBeyond <- function(vString){
        
        sPattern <- '(((\\s){1,20})?("\\{)?(\\s){1,20}\"\"longitude\"\"(.*))|(((\\s){1,20})?("\\{)?(\\s){1,20}\"\"latitude\"\"(.*))'
        
        return(gsub(sPattern, "", vString)) 
        
}

removeWeirdPattern <- function(vString){
        
        # Applied after all the post-processing
        # user description,age,weird location pattern
        sPattern <- '.{1,150},[0-9]{2},\t((\\s)?[A-Za-z][a-zA-Z]{1,15}(\\s)?([A-Z][A-Za-z]{1,15})?((,)?(\\s)?)?([A-Z][a-zA-Z.]{1,15})?(\\s[A-Z][A-Za-z]{1,15})?\")'
        
        return(sub(sPattern,"",vString))
}

removeLocationRetired <- function(vString){
        
        #Removes City / Country data from the start of a string        
        
        sPattern <- '^((\\s)?[A-Za-z][a-zA-Z]{1,15}(\\s)?([A-Z][A-Za-z]{1,15})?(,(\\s)?)?([A-Z][a-zA-Z.]{1,15})?(\\s[A-Z][A-Za-z]{1,15})?\")'
        
        return(sub(sPattern,"",vString))
        
}


removeDoubleQuoteTabAndBeyond <- function(vStrings){
        
        sPattern <- '\"\t(.*)'
        
        return(gsub(sPattern,"",vStrings))
}
