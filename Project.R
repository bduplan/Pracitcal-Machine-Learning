library(ggplot2, quietly=TRUE)
library(caret, quietly=TRUE)
library(iterators, quietly=TRUE)
library(parallel, quietly=TRUE)
library(foreach, quietly=TRUE)
library(doParallel, quietly=TRUE)
library(randomForest, quietly=TRUE)
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

setwd("C:/Users/bdupl/Documents/R/8 Practical Machine Learning/Project")

## Import CSV Data
training <- read.csv("pml-training.csv")
validation <- read.csv("pml-testing.csv")

## Remove columns that are heavily NA in either training or testing data
propNA <- function(x){sum(is.na(x))/length(x)}
propNA.train <- apply(training, 2, propNA)
propNA.val <- apply(validation, 2, propNA)
propNA.keep <- propNA.train < 0.80 & propNA.val < 0.8

training <- training[, propNA.keep]
validation <- validation[, propNA.keep]

## Remove columns that are irrelevant
irrel <- grep("time|_window|user_name", names(training))
training <- training[, -irrel]
training <- training[, -1] #Get rid of index ("X")

## Make classe a factor variable
# training$classe <- as.factor(training$classe)

## Convert factor variables to numeric
facCol <- which(sapply(training, is.factor))
facCol <- facCol[-length(facCol)] #Omit the classe factor from conversion
training[, facCol] <- sapply(training[, facCol], as.numeric)

## Breaking up training into training and testing
set.seed(987)
seltrain <- createDataPartition(y=training$classe, p = 0.6, list = FALSE)
train <- training[seltrain, ]
test <- training[-seltrain, ]

## Fit an LDA model to the training data
opts <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
LDAfit <- train(classe ~ ., data = train, method = "lda", trControl = opts)
predLDA <- predict(LDAfit, train)
confusionMatrix(predLDA,train$classe)

## Fit a Random Forest Model to the training data
if(file.exists("RFfit.rda")) {
    load("RFfit.rda")
} else {
    opts <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
    RFfit <- train(classe ~ ., data = train, method = "rf", trControl = opts)
    save(RFfit, file = "RFfit.rda")
}

predRF <- predict(RFfit, train)
confusionMatrix(predRF, train$classe)

## Plot the importance of each variable
varImpPlot(RFfit$finalModel,
           main="Variable Importance Plot: Random Forest",
           type=2)

plot(RFfit,
     main = "Accuracy by Number of Predictors")
plot(RFfit$finalModel,
     main = "Accuracy by Number of Folds")

## Predict on the test dataset
predRF_test <- predict(RFfit, test)
confusionMatrix(predRF_test, test$classe)

## Predict on the validation dataset
predRF_val <- predict(RFfit, validation)
print(predRF_val)

stopCluster(cluster)





