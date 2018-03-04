if (! "tm" %in% installed.packages()){
    install.packages("tm", repos = "https://cloud.r-project.org/")
}
library(tm)
library(SnowballC)
library(wordcloud)
sms <- read.csv("../spam.csv", stringsAsFactors = FALSE, fileEncoding="latin1")[,1:2]
sms$v1 <- as.factor(sms$v1)
names(sms) <- c("Label", "Text")

str(sms)

## data preparation
sms_corpus <- Corpus(VectorSource(sms$Text))
print(sms_corpus)

## clean redundant stuff
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

## 
corpus_clean <- Corpus(VectorSource(corpus_clean))
dtm <- DocumentTermMatrix(corpus_clean)


## split training and testing
## I want approximately equal proportion of ham/spam in training and testing 
indHam <- sample(which(sms$Label == "ham"), 0.8 * sum(sms$Label == "ham"), 
                   replace = FALSE)
indSpam <- sample(which(sms$Label == "spam"), 0.8 * sum(sms$Label == "spam"), 
                  replace = FALSE)
indTrain <- c(indHam, indSpam)
indTest <- setdiff(1:nrow(sms), indTrain)
## split data
datTrain <- sms[indTrain, ]
datTest <- sms[indTest, ]

## split the corpus
corpusTrain <- corpus_clean[indTrain]
corpusTest <- corpus_clean[indTest]

## split the document-term matrix
dtmTrain <- DocumentTermMatrix(corpusTrain)
dtmTest <- DocumentTermMatrix(corpusTest)

## wordcloud for spam and ham, separately
spam <- subset(datTrain, Label == "spam")
ham <- subset(datTest, Label == "ham")
wordcloud(spam$Text, max.words = 40)
wordcloud(ham$Text, max.words = 40)

## frequent words
freqTerms <- findFreqTerms(dtmTrain, 3)
reducedDtmTrain <- DocumentTermMatrix(corpusTrain, list(dictionary = freqTerms))
reducedDtmTest <-  DocumentTermMatrix(corpusTest, list(dictionary = freqTerms))

