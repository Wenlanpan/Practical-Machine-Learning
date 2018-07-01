# Practical-Machine-Learning
## HTML
https://winnie97.github.io/Practical-Machine-Learning/
## Background
Using devices such as <i>Jawbone Up, Nike FuelBand, and Fitbit</i> it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Data Preprocessing
Load required package
```r
library(caret)
library(rpart）
library(rpart.plot)
library(randomForest)
library(corrplot)
library(rattle)
library(gbm)
```
Load the same seed
```r
set.seed(1)
```
## Data Processing
### Download Data
```r
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```
### Read Data
```r
training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))
```
### Clean Data
Remove columns that contain NA missing values
```r
training <- training[,colSums(is.na(training))==0]
testing <- testing[,colSums(is.na(testing))==0]
```
Remove columns of non-accelerometer measurements
```r
classe <- training$classe
trainingRemoved <- grepl("^X|timestamp|window", names(training))
training <- training[,!trainingRemoved]
trainingCleaned <- training[, sapply(training, is.numeric)]
trainingCleaned$classe <- classe
testingRemoved <- grepl("^X|timestamp|window", names(testing))
testing <- testing[,!testingRemoved]
testingCleaned <- testing[, sapply(testing, is.numeric)]
```
Remove NearZeroVariance
```r
nzv <- nearZeroVar(trainingCleaned, saveMetrics=TRUE)
trainingCleaned <- trainingCleaned[,nzv$nzv==FALSE]
nzv <- nearZeroVar(testingCleaned,saveMetrics=TRUE)
testingCleaned <- testingCleaned[,nzv$nzv==FALSE]
```
Cleaned training set: 19622 observations and 53 variables<br>
Cleaned testing set: 20 observations and 53 variables
### Portion Data
Splicing training data set into 2 data sets, 70% for myTraining, 40% for myTesting
```r
inTrain <- createDataPartition(y=trainingCleaned$classe, p=0.7, list=FALSE)
myTraining <- trainingCleaned[inTrain,]
myTesting <- trainingCleaned[-inTrain,]
```
## Data Modeling
3 different model algorithms were tested

Random forest decision trees (rf)<br>
Decision trees with CART (rpart)<br>
Gradient Boosting trees (gbm)<br>
Cross Validation: 5-fold<br>
```r
fitControl <- trainControl(method="cv",number=5)
modelRF <- train(
classe ~.,
data=myTraining,
trControl=fitControl,
method="rf",
ntree=100
)
save(modelRF, file="./ModelFitRF.RData")
modelCART <- train(
classe ~.,
data=myTraining,
trControl=fitControl,
method="rpart"
)
save(modelCART, file="./ModelFitCART.RData")
modelGBM <- train(
classe ~.,
data=myTraining,
trControl=fitControl,
method="gbm"
)
save(modelGBM, file="./ModelFitGBM.RData")

```
## Model Validation (Out-of-sample error)
```r
predictRF <- predict(modelRF, newdata=myTesting)
cmRF <- confusionMatrix(predictRF, myTesting$classe)
predictCART <- predict(modelCART, newdata=myTesting)
cmCART <- confusionMatrix(predictCART, myTesting$classe)
predictGBM <- predict(modelGBM, newdata=myTesting)
cmGBM <- confusionMatrix(predictGBM, myTesting$classe)
AccuracyResults <- data.frame(
Model=c("RF","CART","GBM"),
Accuracy=rbind(cmRF$overall[1],cmCART$overall[1],cmGBM$overall[1])
)
print(AccuracyResults)
```
````
  Model  Accuracy
1    RF 0.9950722
2  CART 0.4910790
3   GBM 0.9658454
````
The most accurate model is Random Forest
```r
cmRF
```
````
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1672    6    0    0    0
         B    1 1130    5    0    2
         C    1    2 1018    4    0
         D    0    1    3  958    2
         E    0    0    0    2 1078

Overall Statistics
                                          
               Accuracy : 0.9951          
                 95% CI : (0.9929, 0.9967)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9938          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9988   0.9921   0.9922   0.9938   0.9963
Specificity            0.9986   0.9983   0.9986   0.9988   0.9996
Pos Pred Value         0.9964   0.9930   0.9932   0.9938   0.9981
Neg Pred Value         0.9995   0.9981   0.9984   0.9988   0.9992
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2841   0.1920   0.1730   0.1628   0.1832
Detection Prevalence   0.2851   0.1934   0.1742   0.1638   0.1835
Balanced Accuracy      0.9987   0.9952   0.9954   0.9963   0.9979

````
## Model Testing
```r
result <- predict(modelRF, testingCleaned[, -length(names(testingCleaned))])
result
```
````
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
````
