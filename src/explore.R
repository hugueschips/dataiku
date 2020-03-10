##################################################
# Author: Daniel Durrenberger
# Company: Dataiku
# Object: Test
# Date: 05.03.2020
# 
######################################

library(tidyverse)
library(dplyr)
library(dbplyr)
library(xgboost)
library(tibble)
library(ggplot2)
library(pROC)
library(nnet)
library(caret)
library(corrplot)

#################### Wether or not you want to train again or load results
trainAgain <- TRUE

#################### Choice of learning algorith
######################## 2 options are provided : xgboost or neural network
method = 'xgbTree'
# method = 'nnet'

####################  Load Files
trainFile <- "./data/census_income_learn.csv"
testFile <- './data/census_income_test.csv'
trainDataRaw <- read.csv(file = trainFile, header = FALSE) %>% as.tibble
testDataRaw <- read.csv(file = testFile, header = FALSE) %>% as.tibble
trainDataRaw %>% head #See what it looks like

####################  Clean data
######################## Convert all non numeric into dummy variables 
######################## and keep numerics unchanged
customPreprocess2 <- function(trainDataRaw) {
  trainDataClean <- trainDataRaw %>%
    mutate(class = if_else(as.numeric(V42)==2, 'over50', 'below50') %>% factor) %>%
    select(-V42) %>%
    as.tibble
  dummies <- dummyVars(class ~ ., data = trainDataClean) 
  trainDataDummies <- predict(dummies, newdata = trainDataClean) %>% 
    as.tibble
  trainDataDummies$class = trainDataClean$class %>% factor
  trainDataDummies
}

allData <- bind_rows(
  trainDataRaw %>% mutate(train = 1),
  testDataRaw %>% mutate(train = 0)) %>%
  customPreprocess2

######################## Remove columns with near zero variance
######################## necessary for many algorithms
######################## it is not needed by algorithms like xgboost
######################## but it speeds up the training
nzv <- nearZeroVar(allData)
allData <- allData[,-nzv]

allData %>% head #See result

##################### Split data
######################## One half of training set is used to train
xytrain <- allData %>%
  filter(train == 1) %>%
  select(-train) %>%
  group_by(class) %>%
  sample_frac(.5) %>%
  ungroup

######################## The other half for calibration
xycal <- allData %>%
  filter(train == 1) %>%
  select(-train) %>%
  anti_join(xytrain) %>%
  ungroup

######################## Test set is left untouched
xytest <- allData %>%
  filter(train == 0) %>%
  select(-train) %>%
  ungroup

######################## Check class balance
xytrain$class %>% summary
xycal$class %>% summary
xytest$class %>% summary
xytrain %>% dim

######################## Produce correlation matrix of train set
############################## (simplify names to plot something nice)
xytrain2 <- xytrain %>% select_if(is.numeric)
colnames(xytrain2) <- paste0('V', 1:ncol(xytrain2))
correlationMatrix <- cor(xytrain2)
print(correlationMatrix)
################################## summarize the correlation matrix
corrplot(correlationMatrix)
################################## find attributes that are highly
################################## corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.80)
################################## print indexes of highly correlated attributes
print(highlyCorrelated)
################################## print names of highly correlated attributes
featureNames <- colnames(xytrain %>% select_if(is.numeric))
highlyCorrelatedNames <- featureNames[highlyCorrelated]
print(highlyCorrelatedNames)
rm(xytrain2)

#################### Remove higly correlated features
xytrain <- xytrain %>% select(-highlyCorrelatedNames)
xycal <- xycal %>% select(-highlyCorrelatedNames)
xytest <- xytest %>% select(-highlyCorrelatedNames)

#################### Train Model
######################## Settings
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1, 
  summaryFunction = twoClassSummary,
  classProbs = TRUE, 
  verboseIter = TRUE)

######################## Training
if (trainAgain) {
fit <- train(
  x = xytrain %>% select(-class) %>% as.matrix,
  y = xytrain$class,
  tuneLength = 5,
  method = method,
  trControl = fitControl,
  metric = 'ROC',
  verbose = TRUE
)
fit$results %>% arrange(-ROC) %>% head
saveRDS(fit, file = 'model_new.RData')
}
##################### Results
######################## Save and load training
if (!(trainAgain)) {fit <- readRDS('model_xgboost.RData')}

######################## Importance of variables
############################ Here feature names are not helping, 
############################ but usually, it is a very safe way to evaluate the model
############################ as the most important vars should be known and studied
varImp(fit, scale = TRUE) 

######################## Calibration
############################# The aim in this section is to set the threshold 
############################# between both of the classes. The defaut 0.5 probability
############################# may not be the best choice. The decision should be taken
############################# considering kappa, sensitivity, specificity, ROC curve
############################# Accuracy should not be considered as the dataset is imbalanced
######## Predict calibration set
calPredict <- predict(fit, 
                            newdata = xycal %>% select(-class) %>% as.matrix, 
                            type = 'prob') %>%
  mutate(pred = if_else(below50 > .5, 'below50', 'over50') %>% factor) %>%
  mutate(obs = xycal$class %>% factor)

############ ROC curve for calibration set
pROC_obj_cal <- roc(calPredict$obs, calPredict$below50, ci = TRUE)
pROC_obj_cal
plot(pROC_obj_cal)
############# Set threshold using ROC curve
threshold <- 0.8
calPredict <- calPredict %>%
  mutate(pred = if_else(below50 > threshold, 'below50', 'over50') %>% factor)
############# See confusion matrix with set threshold
confusionMatrix(calPredict$obs, calPredict$pred)

######################## Test
############################# Test set consist of untouched data, independant to training
############################# or calibration to assess the behavior the model with 
############################# new unknown data
########### Predict test set with chosen threshold
testPredict <- predict(fit, 
                      newdata = xytest %>% select(-class) %>% as.matrix, 
                      type = 'prob') %>%
  mutate(pred = if_else(below50 > threshold, 'below50', 'over50') %>% factor) %>%
  mutate(obs = xytest$class %>% factor)
########### See confusion matrix
confusionMatrix(testPredict$obs, testPredict$pred)
########### and ROC curve
pROC_obj_cal <- roc(testPredict$obs, testPredict$below50, ci = TRUE)
pROC_obj_cal
plot(pROC_obj_cal)





