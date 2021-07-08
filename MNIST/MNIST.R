library(tidyverse)
library(dslabs)
mnist <- read_mnist()

set.seed(1990)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])


## PREPROCESSING
library(matrixStats) 
sds <- colSds(x) 
qplot(sds, bins = 256) 

library(caret)  
nzv <- nearZeroVar(x) 

image(matrix(1:784 %in% nzv, 28, 28))

col_index <- setdiff(1:ncol(x), nzv) 
length(col_index)

#naming columns in train and test sets
colnames(x) <- 1:ncol(mnist$train$images)  
colnames(x_test) <- colnames(x) 


## KNN

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[ ,col_index], y, 
                   method = "knn", 
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)
train_knn

#fit knn
fit_knn <- knn3(x[, col_index], y,  k = 3)

#test set prediction and accuracy
y_hat_knn <- predict(fit_knn, x_test[, col_index], type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]


## RANDOM FOREST
library(randomForest)
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))

train_rf <-  train(x[, col_index], y, 
                   method = "rf", 
                   ntree = 150,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)

#fit the rf model
fit_rf <- randomForest(x[, col_index], y, minNode = train_rf$bestTune$mtry)

#check if I ran enough trees
plot(fit_rf)

#predict on test set and check accuracy
y_hat_rf <- predict(fit_rf, x_test[ ,col_index])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]


## VARIABLE IMPORTANCE
imp <- importance(fit_rf) 

mat <- rep(0, ncol(x))  
mat[col_index] <- imp  
image(matrix(mat, 28, 28)) 


## ENSEMBLES

p_rf <- predict(fit_rf, x_test[,col_index], type = "prob")  
p_rf <- p_rf / rowSums(p_rf)  

p_knn <- predict(fit_knn, x_test[,col_index])  

p <- (p_rf + p_knn)/2  

y_pred <- factor(apply(p, 1, which.max)-1)  

confusionMatrix(y_pred, y_test)$overall["Accuracy"]  
