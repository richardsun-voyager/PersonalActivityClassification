---
title: "MachineLearningProject1"
author: "RichardSun"
date: "Wednesday, April 15, 2015"
output: html_document
---

##Summary
This report aims to analyze the relationships between personal activities and the types of the results.Considering the size of features, we select original observation variables, and apply LDA ,QDA, tree decision models to train the fits.QDA model does better on this problem, also we give the types of test data set based on QDA model.

##Get and clean the data
Download the original data online.

```r
setwd("E:\\My Document\\ËïµÄÎÄµµ\\Coursera Assignments\\DataScience\\MachineLearning")
url1="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url2="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
setInternet2(use=TRUE)
training<-read.csv(url1)
testing<-read.csv(url2)
dim(training)
```

```
## [1] 19622   160
```
In the training set, there are 19622 rows and 160 columns, that means 160 features for this data.Let's take a look at the features.

```r
names(training)
```
It can be seen that many variables are descriptive, according to the instruction,our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell.Now we select these variables.

```r
variables<-names(training)
##index<-grep("accel",variables)
activities<-c("forearm","arm","belt","dumbbell")
index<-NULL
for(activity in activities)
  {
  index<-c(index,grep(activity,variables,fixed=TRUE))
  }
index=unique(index)
cols<-c(variables[index],"classe")
trainingNew<-training[,cols]
dim(trainingNew)
```

```
## [1] 19622   153
```
There are still 153 variables, a quite large number for prediction.We need think about the predictors before we do training.

##Select the model
1.Select the predictors
According to Qualitative Activity Recognition of Weight Lifting Exercises, they selected 17 features based on backtracking. In my opinion, these variables are statistical values , they are derived features, as they have been discussed by that paper, we'd better try another way, to explore the original features,test how those original variables work on predictions.There are many methods such as forward, backward, shrinkage for selecting predictors, allowing for the calculation capability of my laptop, we make use all these predictors at first.

```r
starts<-c("total","var","std","kurtosis","avg","skewness","max","min","amplitude")
index<-NULL
for(start in starts)
  {
  index<-c(index,grep(start,names(trainingNew),fixed=TRUE))
  }
index<-unique(index)
trainingNew<-trainingNew[,-index]
dim(trainingNew)
```

```
## [1] 19622    49
```
Now there are 48 features as predictors, take a look at them.

```r
sum(is.na(trainingNew))
```

```
## [1] 0
```

```r
summary(trainingNew$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
No missing values inside, so we need not omit them. Also there are five types for classe, therefore, t's not feasible to apply "glm" method.

2.Linear Discriminant Analysis
It is a classification problem, there are many methods for classifying.Let's begin with the most simple one, linear discriminant analysis, suppose the relationship between the predictors and the classe is linear, we divide the training set into two parts, one is for training and the other is for validation.

```r
library(caret);set.seed(111);
library(MASS)
inTrain<-createDataPartition(y=trainingNew$classe,p=0.75,list=FALSE)
trainingSet<-trainingNew[inTrain,]
testingSet<-trainingNew[-inTrain,]
modFit1<-train(classe~.,data=trainingSet,method="lda")
testClasse<-predict(modFit1,trainingSet)
confusionMatrix(testClasse,trainingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3413  437  276  148  123
##          B  143 1812  236  183  451
##          C  300  328 1704  241  223
##          D  299  113  290 1723  282
##          E   30  158   61  117 1627
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6984          
##                  95% CI : (0.6909, 0.7058)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6181          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8155   0.6362   0.6638   0.7143   0.6013
## Specificity            0.9066   0.9147   0.9101   0.9200   0.9695
## Pos Pred Value         0.7762   0.6414   0.6094   0.6365   0.8164
## Neg Pred Value         0.9252   0.9129   0.9276   0.9426   0.9152
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2319   0.1231   0.1158   0.1171   0.1105
## Detection Prevalence   0.2987   0.1919   0.1900   0.1839   0.1354
## Balanced Accuracy      0.8611   0.7754   0.7870   0.8172   0.7854
```
The accuracy for training part is almost 0.7, not so bad,next check the accuracy for testing part.

```r
testClasse<-predict(modFit1,testingSet)
confusionMatrix(testClasse,testingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1123  154  108   52   38
##          B   36  582   82   64  143
##          C  114  109  535   83   53
##          D  110   46  105  574  108
##          E   12   58   25   31  559
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6878          
##                  95% CI : (0.6746, 0.7008)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6047          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8050   0.6133   0.6257   0.7139   0.6204
## Specificity            0.8997   0.9178   0.9113   0.9100   0.9685
## Pos Pred Value         0.7614   0.6417   0.5984   0.6087   0.8161
## Neg Pred Value         0.9207   0.9082   0.9202   0.9419   0.9189
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2290   0.1187   0.1091   0.1170   0.1140
## Detection Prevalence   0.3008   0.1850   0.1823   0.1923   0.1397
## Balanced Accuracy      0.8524   0.7656   0.7685   0.8120   0.7945
```
This model goes well with the testing data too, but the accuracy is not sufficient. We need improve.We need reconsider the predictors,some of them may be correlated,let's check out the correlations.
![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

The correlation result above proves our assumptions, some variables are really correlated, so we adopt PCA preprocessing method to reduce numbers of predictors.

```r
modFit2<-train(classe~.,data=trainingSet,preProcess="pca",method="lda")
testClasse<-predict(modFit2,trainingSet)
confusionMatrix(testClasse,trainingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2829  686  668  169  396
##          B  360 1241  276  401  500
##          C  346  424 1346  339  331
##          D  554  397  194 1229  317
##          E   96  100   83  274 1162
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5304          
##                  95% CI : (0.5223, 0.5385)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4036          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6760  0.43574  0.52435   0.5095  0.42942
## Specificity            0.8178  0.87051  0.88149   0.8812  0.95396
## Pos Pred Value         0.5958  0.44672  0.48313   0.4567  0.67755
## Neg Pred Value         0.8640  0.86541  0.89767   0.9016  0.88126
## Prevalence             0.2843  0.19350  0.17441   0.1639  0.18386
## Detection Rate         0.1922  0.08432  0.09145   0.0835  0.07895
## Detection Prevalence   0.3226  0.18875  0.18929   0.1828  0.11652
## Balanced Accuracy      0.7469  0.65313  0.70292   0.6954  0.69169
```
The accuracy gets worse, PCA is not proper in this case.

3.Use QDA method
The number of observations is much more than that of features, we have sufficient training groups, according to the the book An Introduction to Statistical Learning, quadratic discriminant analysis is more feasible.

```r
modFit3<-train(classe~.,data=trainingSet,method="qda")
testClasse<-predict(modFit3,trainingSet)
confusionMatrix(testClasse,trainingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3992  207    6   14    2
##          B  132 2349  121    9   83
##          C   26  269 2413  345  113
##          D   24    8   16 2018   66
##          E   11   15   11   26 2442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8978          
##                  95% CI : (0.8928, 0.9027)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8707          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9539   0.8248   0.9400   0.8367   0.9024
## Specificity            0.9783   0.9709   0.9380   0.9907   0.9948
## Pos Pred Value         0.9457   0.8719   0.7622   0.9465   0.9749
## Neg Pred Value         0.9816   0.9585   0.9867   0.9687   0.9784
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2712   0.1596   0.1639   0.1371   0.1659
## Detection Prevalence   0.2868   0.1830   0.2151   0.1449   0.1702
## Balanced Accuracy      0.9661   0.8979   0.9390   0.9137   0.9486
```
It seems QDA method performs better on the prediction.Let's check the testing data set.

```r
testClasse<-predict(modFit3,testingSet)
confusionMatrix(testClasse,testingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1335   79    2   10    1
##          B   38  743   43    4   21
##          C    7  118  801  118   26
##          D   10    3    7  664   17
##          E    5    6    2    8  836
## 
## Overall Statistics
##                                          
##                Accuracy : 0.8929         
##                  95% CI : (0.884, 0.9015)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.8645         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9570   0.7829   0.9368   0.8259   0.9279
## Specificity            0.9738   0.9732   0.9336   0.9910   0.9948
## Pos Pred Value         0.9355   0.8751   0.7486   0.9472   0.9755
## Neg Pred Value         0.9827   0.9492   0.9859   0.9667   0.9839
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2722   0.1515   0.1633   0.1354   0.1705
## Detection Prevalence   0.2910   0.1731   0.2182   0.1429   0.1748
## Balanced Accuracy      0.9654   0.8781   0.9352   0.9084   0.9613
```
The accuracy is quite satisfying for the testing data.

4.Tree Decisions
Aside from two regression methods above, I'd like to apply a tree-decision method to this classifying problem.Tree decisions are easy to interpret.Let's have a look at this model.Rpart is introduced here.

```r
library(rpart)
modFit4<-train(classe~.,data=trainingSet,method="rpart")
testClasse<-predict(modFit4,trainingSet)
confusionMatrix(testClasse,trainingSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3813 1188 1181 1081  382
##          B   68  971   85  433  372
##          C  296  689 1301  898  714
##          D    0    0    0    0    0
##          E    8    0    0    0 1238
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4976          
##                  95% CI : (0.4894, 0.5057)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3434          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9111  0.34094   0.5068   0.0000  0.45750
## Specificity            0.6362  0.91929   0.7863   1.0000  0.99933
## Pos Pred Value         0.4988  0.50337   0.3338      NaN  0.99358
## Neg Pred Value         0.9474  0.85323   0.8830   0.8361  0.89103
## Prevalence             0.2843  0.19350   0.1744   0.1639  0.18386
## Detection Rate         0.2591  0.06597   0.0884   0.0000  0.08411
## Detection Prevalence   0.5194  0.13106   0.2648   0.0000  0.08466
## Balanced Accuracy      0.7737  0.63012   0.6465   0.5000  0.72842
```
The accruracy of tree-decisions method is not disappointing.Consequently, we select QDA model as our model for this project. Next,we figure out the train and test errors of QDA model.

##Cross Validation
Due to the large size of the training dataset, it is impossible to apply leave-one-out-validation on my laptop.Therefore,we select k-Fold-Cross-Validation.We randomly divide the training dataset into k groups or folds.Assume one fold is treated as a validation set, then the model fit on the rest k-1 folds.Here we define error rate as 1-accuracy for each fold.Suppose k=10.

```r
set.seed(222)
fold<-(round(dim(trainingNew)[1]/10)-1)
kIndex<-sample(dim(trainingNew)[1],dim(trainingNew)[1])
trainingError=0
testingError=0
for(i in 1:10)
  {
  if(i<10){inTest<-kIndex[(fold*(i-1)+1):(fold*i)]}
  else    {inTest<-kIndex[-(1:(fold*9))]}
  trainingSet<-trainingNew[-inTest,]
  testingSet<-trainingNew[inTest,]
  modFit<-train(classe~.,data=trainingSet,method="qda")
  Classe<-predict(modFit,trainingSet)
  temp1<-confusionMatrix(Classe,trainingSet$classe)
  trainingError[i]=1-temp1[[3]][1]
  Classe<-predict(modFit,testingSet)
  temp2<-confusionMatrix(Classe,testingSet$classe)
  testingError[i]=1-temp2[[3]][1]
  }
```
Take a look at the error rates.

```r
par(mfrow=c(2,1))
plot(trainingError);plot(testingError)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png) 
From the plots, it can be seen the approximation of test error is quite stable and low.
The expected output error rate should be the average of the testing error rates.

```r
mean(trainingError);sd(trainingError)
```

```
## [1] 0.1068471
```

```
## [1] 0.001490303
```

```r
mean(testingError);sd(testingError)
```

```
## [1] 0.1138446
```

```
## [1] 0.007228176
```


##Predict the given testing data 
According to the training results, We use QDA model to predict the test cases.

```r
testing$classe<-NA
testingNew<-testing[,names(trainingNew)]
inTest<-kIndex[(fold*(6-1)+1):(fold*6)]
trainingSet<-trainingNew[-inTest,]  
modFit<-train(classe~.,data=trainingSet,method="qda")
testClasse<-predict(modFit,testingNew)
testClasse
```

```
##  [1] A A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


##Conclusion
As for a large size of variable classifying problem, first we select original observation features to predict the outcome,also we apply PCA to reduce the dimension;then,we try LDA , QDA, Rpart models to train the fit respectly, find QDA leads to a better performance for this data set;next, we use k-folds cross validation to test the error rates, actually this QDA model is quite stable;finally, we run this model on the test data set, and predict the classe values of each item. 


