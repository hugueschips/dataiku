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
library(earth)
library(nnet)
library(rocc)
library(caret)

# Project metadata
method = 'xgbTree'
# method = 'mlpKerasDecayCost'
# method = 'nnet'
# method = 'rocc'

# Files
trainFile <- "./data/census_income_learn.csv"
testFile <- './data/census_income_test.csv'

# Load Files
trainDataRaw <- read.csv(file = trainFile, header = FALSE) %>% as.tibble
testDataRaw <- read.csv(file = testFile, header = FALSE) %>% as.tibble
trainDataRaw %>% head #See what it looks like

# Clean data
## Convert all non numeric into dummy variables and keep numerics unchanged
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

# Remove columns with near zero variance.
nzv <- nearZeroVar(allData)
allData <- allData[,-nzv]

allData %>% head #See result

# trainDataClean <- trainDataRaw %>% customPreprocess2
# testDataClean <- testDataRaw %>% customPreprocess2



# Explore data
# summary(trainDataClean$X73)
# snippet <- trainDataClean %>% 
#                         sample_n(100) %>%
#                         select_if(is.numeric)
# graph1 <- ggplot(trainDataClean, aes(x=X73, y=over50k)) + geom_bar()
# graph1
# 
# 
# colNames <- colnames(trainDataClean)
# ggplot(trainDataClean %>% sample_n(1000), aes(x=X0.6, y=as.numeric(class)))+
#   # geom_point(size=2, alpha=0.4)+
#   stat_smooth(method="loess", colour="blue", size=1.5)+
#   # xlab("Age")+
#   ylab("Probability of earning over 50K")+
#   theme_bw()



# Split data
## One half of training set is used to train
xytrain <- allData %>%
  filter(train == 1) %>%
  select(-train) %>%
  group_by(class) %>%
  sample_frac(.5) %>%
  ungroup

## The other half for calibration
xycal <- allData %>%
  filter(train == 1) %>%
  select(-train) %>%
  anti_join(xytrain) %>%
  ungroup

## Test set is left untouched
xytest <- allData %>%
  filter(train == 0) %>%
  select(-train) %>%
  ungroup

## Check class balance
xytrain$class %>% summary
xycal$class %>% summary
xytest$class %>% summary
xytrain %>% dim

# Train Model
## Settings
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1, 
  summaryFunction = twoClassSummary,
  classProbs = TRUE, 
  verboseIter = TRUE)

## training
fit <- train(
  x = xytrain %>% select(-class) %>% as.matrix,
  y = xytrain$class,
  tuneLength = 5,
  method = method,
  trControl = fitControl,
  metric = 'ROC',
  verbose = TRUE
)

# Results
## Training performance
fit$results %>% arrange(-ROC) %>% head

## Importance of variables
varImp(fit, scale = TRUE) 

# Calibration
calPredict <- predict(fit, 
                            newdata = xycal %>% select(-class) %>% as.matrix, 
                            type = 'prob') %>%
  mutate(pred = if_else(below50 > .5, 'below50', 'over50') %>% factor) %>%
  mutate(obs = xycal$class %>% factor)

threshold <- 0.6

calPredict <- calPredict %>%
  mutate(pred = if_else(below50 > threshold, 'below50', 'over50') %>% factor)
confusionMatrix(calPredict$obs, calPredict$pred)

pROC_obj_cal <- roc(calPredict$obs, calPredict$below50, ci = TRUE)
pROC_obj_cal
plot(pROC_obj_cal)

# Test
testPredict <- predict(fit, 
                      newdata = xytest %>% select(-class) %>% as.matrix, 
                      type = 'prob') %>%
  mutate(pred = if_else(below50 > .8, 'below50', 'over50') %>% factor) %>%
  mutate(obs = xytest$class %>% factor)

testPredict <- testPredict %>%
  mutate(pred = if_else(below50 > threshold, 'below50', 'over50') %>% factor)
confusionMatrix(testPredict$obs, testPredict$pred)

pROC_obj_cal <- roc(testPredict$obs, testPredict$below50, ci = TRUE)
pROC_obj_cal
plot(pROC_obj_cal)


saveRDS(fit, file = 'model_nn.RData')


