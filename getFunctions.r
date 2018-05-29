#Contains Get functions that are used when reading in the data for both Retired and Normal Users


getFirstThreeHashKeys <- function(vLines){
        
        #Input: vector of lines
        #Get first three hashed keys  
        #Stores them in data frame and returns the df
        
        hashKeyPattern <- '^(([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,})),'
        
        vHashKeys <- str_extract(vLines, hashKeyPattern)
        vHashKeys <- substr(vHashKeys, 1, str_length(vHashKeys)-1)       
        l         <- str_split(vHashKeys,",", n=3)       
        df        <- do.call(rbind.data.frame, l)
        names(df) <- c("gender", "sender_name", "sender_id")
        
        return(df)
        
}


getFileTitle <- function(vFileRef){
        
        # Removes the filepath, then removes the extensions and returns the title of the file without extension
        # gsub has no effect if paramater if the input is just a fileName
        
        vDirectoryRemoved  <- gsub("^(\\./(.*)/)","",vFileRef)
        vFileTitle         <- gsub("(\\.)[A-Za-z]{3}((\\.)[A-Za-z]{3})?","",vDirectoryRemoved)
                
        return(vFileTitle)
        
}


getFullFilePath <- function (filePath, vFileNames){
        # Return vString of the full path including filenames
        # Note: filePath must be of type "./..../"      
        
        vFileRef <- paste(filePath, vFileNames, sep="")     
        return(vFileRef) 
        
}

getAllSpecialCharPositions <- function (vString,specialChar){
        
        #Returns all special Char positions of a String vector as a list        
        vIndexes <- str_locate_all(vString, specialChar)        
        
        return(vIndexes)
        
}


getRelevantTweetIndices <- function(vTweets){
        
        # Checks which tweets are retweets [not-relevant]
        # Check if tweet is empty [not relevant]
        # Check if tweet is NA
        # Check if tweet has at least 5 chars
        # Return vector with indices that have valid tweets 
        vHasLength5PlusChar <- (str_length(vTweets) >= 5)
        vHasRetweet         <- (removeRetweets(vTweets) == "")        
        vIndexRelevant      <- (vTweets != "" & !is.na(vTweets) & vHasRetweet == FALSE & vHasLength5PlusChar == TRUE)  
        
        return(vIndexRelevant)
}
