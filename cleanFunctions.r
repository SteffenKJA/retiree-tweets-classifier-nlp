#Contains cleaning functions that are used when reading in the data for both Retired and Normal Users


removeFirst3HashKeys <- function(vFullLine){
        
        ##Removes first three hashed keys
        
        return(sub('^(([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,}),([a-zA-Z0-9]{50,})),',"",vFullLine))
        
}


removeHashAndBeyond <- function(vString){
        
        sPattern <- "(,)?[a-zA-Z0-9]{50,}(,(.*))?"
        
        return(gsub(sPattern,"",vString))    
}


removeNumAndHashAndBeyond <- function(vString){
        
        sPattern <- "(,)?[0-9]{2},[a-zA-Z0-9]{50,}(,(.*))?"
        
        return(gsub(sPattern,"",vString))    
}


removeTwitterDevice <- function (vStrings){
        
        #Removes all text referring to the device being used, mobile phone etc. 
        #and beyond, usually last item in record
        
        return(gsub('\"<a href(.*)',"",vStrings))
        
}


removeDatesTimes <- function(vString, sClassification){           
        
        # Removes dates based on format specific to the specific classification tweet data  
        # sClassification can either be "Retired" or "Normal"
        
        if(identical("Retired", sClassification) == TRUE){
                
                sPattern <- "((,)?(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4}(,)?)|((,)?(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4}(,)?)"
                
        } else { # Normal
                
                sPattern <- "(,)?(Mon|Tue|Wed|Thu|Fri|Sat|Sun),(\\s)?[0-9]{2} [A-Z]([a-z])+ ([0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2} \\+[0]{4}(,)?"
                
        }        
        
        return(gsub(sPattern,"",vString))
}


removeRetweets <- function(vString){
        
        sPattern <- "(.*)RT @(.*)"
        
        return(gsub(sPattern,"",vString))
}


subCharBySpace <- function(vStrings,sChar){
        
        return(gsub(sChar," ",vStrings))
}


