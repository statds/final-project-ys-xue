pkgs <- c("tm", "e1071", "gmodels")
for(i in 1:length(pkgs)){
    if (! pkgs[i] %in% installed.packages()){
        install.packages(pkgs[i], repos = "https://cloud.r-project.org/")
    }
}

library(tm)
library(gmodels)
library(e1071)
library(SnowballC)
library(wordcloud)
sms <- read.csv("../spam.csv", stringsAsFactors = FALSE, encoding = "latin1")[,1:2]
sms$v1 <- factor(sms$v1)
names(sms) <- c("Label", "Text")

str(sms)

## data preparation
sms_corpus <- VCorpus(VectorSource(sms$Text))
print(sms_corpus)

## clean redundant stuff
corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

## 
corpus_clean <- Corpus(VectorSource(corpus_clean))
dtm <- DocumentTermMatrix(corpus_clean)


## split training and testing

datTrain <- sms[1:4200, ]
datTest <- sms[4201:5572, ]

## labels
labelTrain <- sms$Label[1:4200]
labelTest <- sms$Label[4201:5572]

## split the document-term matrix
dtmTrain <- dtm[1:4200, ]
dtmTest <- dtm[4201:5572, ]

## wordcloud for spam and ham, separately
spam <- subset(datTrain, Label == "spam")
ham <- subset(datTest, Label == "ham")
wordcloud(spam$Text, max.words = 40)
wordcloud(ham$Text, max.words = 40)

## frequent words
freqTerms <- findFreqTerms(dtmTrain, 5)
reducedDtmTrain <- dtmTrain[, freqTerms]
reducedDtmTest <-  dtmTest[, freqTerms]

## convert 0, 1 in DTM to yes/no factor, and apply to reduced matrices

convertCount <- function(x){
    x <- factor(x > 0, levels = c(0, 1), labels = c("No", "Yes"))
    return(x)
}

reducedDtmTrain <- apply(reducedDtmTrain, 2, convertCount)
reducedDtmTest <- apply(reducedDtmTest, 2, convertCount)

smsClassifier <- naiveBayes(reducedDtmTrain, labelTrain)
smsPred <- predict(smsClassifier, reducedDtmTest)

CrossTable(smsPred, datTest$Label, prop.chisq = FALSE, prop.t = FALSE, prop.c = TRUE, 
           prop.r = TRUE,dnn = c("predicted", "actual"))
