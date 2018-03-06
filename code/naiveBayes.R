pkgs <- c("tm", "e1071", "gmodels", "stringr", "SnowballC", "wordcloud", "caret")
for(i in 1:length(pkgs)){
    if (! pkgs[i] %in% installed.packages()){
        install.packages(pkgs[i], repos = "https://cloud.r-project.org/")
    }
}

library(tm)
library(stringr)
library(gmodels)
library(e1071)
library(SnowballC)
library(wordcloud)
library(ggplot2)
library(caret)
sms <- read.csv("../spam.csv", stringsAsFactors = FALSE, encoding = "latin1")[,1:2]
sms$v1 <- factor(sms$v1)
names(sms) <- c("Label", "Text")

ggplot(sms, aes(x = Label, fill = Label)) + geom_bar(stat = "count")
sms$Length <- str_length(sms$Text)
ggplot(sms, aes(x = Length, fill = Label)) + geom_histogram(binwidth = 5)

## clean redundant stuff
## data preparation
smsCorpus <- VCorpus(VectorSource(sms$Text))

corpusClean <- tm_map(smsCorpus, removeWords, stopwords())
corpusClean <- tm_map(corpusClean, stripWhitespace)
corpusClean <- tm_map(smsCorpus, content_transformer(tolower))
corpusClean <- tm_map(corpusClean, removeNumbers)
corpusClean <- tm_map(corpusClean, removePunctuation)
corpusClean <- tm_map(corpusClean, stemDocument)

## 
dtm <- DocumentTermMatrix(corpusClean)

## split training and testing
set.seed(1234)
indHam <- sample(which(sms$Label == "ham"), 0.8 * sum(sms$Label == "ham"), 
                                     replace = FALSE)
set.seed(5678)
indSpam <- sample(which(sms$Label == "spam"), 0.8 * sum(sms$Label == "spam"), 
                                     replace = FALSE)
indTrain <- c(indHam, indSpam)
indTest <- setdiff(1:nrow(sms), indTrain)

datTrain <- sms[indTrain, ]
datTest <- sms[indTest, ]
corpusTrain <- corpusClean[indTrain]
corpusTest <- corpusClean[indTest]

## labels
labelTrain <- sms$Label[indTrain]
labelTest <- sms$Label[indTest]

## split the document-term matrix
dtmTrain <- dtm[indTrain,]
dtmTest <- dtm[indTest, ]

freq5 <- findFreqTerms(dtmTrain, 5)
length(freq5)

rdtmTrain <- dtmTrain[, freq5]
rdtmTest <- dtmTest[, freq5]

convertCount <- function(x){
    x <- as.factor(ifelse(x > 0, "yes", "no"))
    return(x)
}

crdtmTrain <- apply(rdtmTrain, 2, convertCount)
crdtmTest <- apply(rdtmTest, 2, convertCount)


smsClassifier <- naiveBayes(crdtmTrain, labelTrain)
smsPred <- predict(smsClassifier, crdtmTest)

CrossTable(smsPred, datTest$Label, prop.chisq = FALSE, prop.t = FALSE, prop.c = TRUE, 
           prop.r = TRUE,dnn = c("predicted", "actual"))

confusionMatrix(smsPred, datTest$Label)
