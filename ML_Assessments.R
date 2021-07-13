library(dslabs) 
library(tidyverse)
library(caret)

data(heights)

y <- heights$sex  
x <- heights$height

set.seed(2007)  
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE) 


install.packages("dslabs")
library(dslabs)
mnist <- read_mnist()


library(dslabs)
library(dplyr)
library(lubridate)
data(reported_heights)

dat <- mutate(reported_heights, date_time = ymd_hms(time_stamp)) %>%
  filter(date_time >= make_date(2016, 01, 25) & date_time < make_date(2016, 02, 1)) %>%
  mutate(type = ifelse(day(date_time) == 25 & hour(date_time) == 8 & between(minute(date_time), 15, 30), "inclass","online")) %>%
  select(sex, type)

y <- factor(dat$sex, c("Female", "Male"))
x <- dat$type

dat %>% filter(type == "inclass") %>% nrow()
dat %>% filter(type == "inclass", sex == "Female") %>% nrow()
dat %>% filter(type == "online") %>% nrow()
dat %>% filter(type == "online", sex == "Female") %>% nrow()

#pred gender based on type
guess_pred <- ifelse(dat$type == "inclass", "Female", "Male")
#accuracy
mean(guess_pred == dat$sex)

#confusion matrix
cm <- table(guess_pred, dat$sex)

sensitivity(cm)
specificity(cm)

#prevalence
sum(dat$sex == "Female")/nrow(dat)


library(caret)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species

set.seed(2, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p=0.5, list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]

#8
find_feature <- function(feature) {
  
  temp_seq <- seq(range(feature)[1], range(feature)[2], by=0.01)
  accuracy <- map_dbl(temp_seq, function(x) {
    y_hat <- ifelse(feature <= x, "versicolor", "virginica") %>%
      factor(levels = levels(train$Species))
    mean(y_hat == train$Species)
  })
  list(accuracy=max(accuracy), cutoff=temp_seq[which.max(accuracy)])
}

find_feature(train$Sepal.Length)

features <- train[-5]
sapply(features, find_feature)

#9
petal_accuracy <- ifelse(test$Petal.Length > 4.7, "virginica", "versicolor")
mean(petal_accuracy == test$Species)

#10
find_feature <- function(feature) {
  
  temp_seq <- seq(range(feature)[1], range(feature)[2], by=0.01)
  accuracy <- map_dbl(temp_seq, function(x) {
    y_hat <- ifelse(feature <= x, "versicolor", "virginica") %>%
      factor(levels = levels(test$Species))
    mean(y_hat == test$Species)
  })
  list(accuracy=max(accuracy), cutoff=temp_seq[which.max(accuracy)])
}

features <- test[-5]
sapply(features, find_feature)

#11
plot(iris, pch=21, bg=iris$Species)

petal_accuracy <- ifelse(test$Petal.Length > 4.7 | test$Petal.Width > 1.5,
                         "virginica", "versicolor")
mean(petal_accuracy == test$Species)


###Conditional Probability

set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
disease <- sample(c(0,1), size=1e6, replace=TRUE, prob=c(0.98,0.02))
test <- rep(NA, 1e6)
test[disease==0] <- sample(c(0,1), size=sum(disease==0), replace=TRUE, prob=c(0.90,0.10))
test[disease==1] <- sample(c(0,1), size=sum(disease==1), replace=TRUE, prob=c(0.15, 0.85))

#What is the probability that an individual has the disease if the test is negative?
mean(disease[test==0])

#If a patient's test is positive, by how many times does that increase their risk of having the disease?
mean(disease[test==1])/mean(disease==1)


#6. Plot conditional prob P(Male | height = x)
library(dslabs)
data("heights")

heights %>% 
  mutate(height = round(height)) %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Male")) %>%
  qplot(height, p, data =.)

#7. This time use the quantile 0.1, 0.2, ..0.9 and the cut() function to assure each group has the same number of points
ps <- seq(0, 1, 0.1)
heights %>% 
  mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(p = mean(sex == "Male"), height = mean(height)) %>%
  qplot(height, p, data =.)


#8. generate data from a bivariate normal distrubution using the MASS package
Sigma <- 9*matrix(c(1,0.5,0.5,1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

ps <- seq(0, 1, 0.1)
dat %>% 
  mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(y = mean(y), x = mean(x)) %>%
  qplot(x, y, data =.)



###Linear Regression

#1. 
library(tidyverse)
library(caret)

set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
n <- 100
Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

set.seed(1, sample.kind="Rounding")
rmse <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

#2. 
mean(rmse)
sd(rmse)

generate_rsme <- function(n) {
  Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
  dat <- MASS::mvrnorm(n = n, c(69, 69), Sigma) %>%
    data.frame() %>% setNames(c("x", "y"))
  
  rmse <- replicate(100, {
    test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    fit <- lm(y ~ x, data = train_set)
    y_hat <- predict(fit, newdata = test_set)
    sqrt(mean((y_hat-test_set$y)^2))
  })
  
  return(list(mean=mean(rmse), sd=sd(rmse)))
}

set.seed(1, sample.kind="Rounding")
n <- c(100, 500, 1000, 5000, 10000)
sapply(n, generate_rsme)


#4. 
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
n <- 100
Sigma <- 9*matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

rmse <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, newdata = test_set)
  sqrt(mean((y_hat-test_set$y)^2))
})

mean(rmse)
sd(rmse)


#6.
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))

fit <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))

fit <- lm(y ~ x_1+x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))


#8.
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.95, 0.75, 0.95, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x_1", "x_2"))

test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_set <- dat %>% slice(-test_index)
test_set <- dat %>% slice(test_index)
fit <- lm(y ~ x_1, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))

fit <- lm(y ~ x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))

fit <- lm(y ~ x_1+x_2, data = train_set)
y_hat <- predict(fit, newdata = test_set)
sqrt(mean((y_hat-test_set$y)^2))


###LOGISTIC REGRESSION

set.seed(2, sample.kind="Rounding") #if you are using R 3.6 or later
make_data <- function(n = 1000, p = 0.5, 
                      mu_0 = 0, mu_1 = 2, 
                      sigma_0 = 1,  sigma_1 = 1){
  
  y <- rbinom(n, 1, p)
  f_0 <- rnorm(n, mu_0, sigma_0)
  f_1 <- rnorm(n, mu_1, sigma_1)
  x <- ifelse(y == 1, f_1, f_0)
  
  test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
  
  list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
       test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()

#Set the seed to 1, then use the make_data() function defined above to generate 25 different datasets with mu_1 <- seq(0, 3, len=25). 
#Perform logistic regression on each of the 25 different datasets (predict 1 if p>0.5) and plot accuracy (res in the figures) vs mu_1 (delta in the figures).
delta <- seq(0, 3, len = 25)
res <- sapply(delta, function(d){
  dat <- make_data(mu_1 = d)
  fit_glm <- dat$train %>% glm(y ~ x, family = "binomial", data = .)
  y_hat_glm <- ifelse(predict(fit_glm, dat$test) > 0.5, 1, 0) %>% factor(levels = c(0, 1))
  mean(y_hat_glm == dat$test$y)
})
qplot(delta, res)


###MATRICES

library(tidyverse)
library(dslabs)
if(!exists("mnist")) mnist <- read_mnist()

class(mnist$train$images)

x <- mnist$train$images[1:1000,] 
y <- mnist$train$labels[1:1000]

grid <- matrix(x[3,], 28, 28)
image(1:28, 1:28, grid)

x <- mnist$train$images
y <- mnist$train$labels

#For each observation in the mnist training data, compute the proportion of pixels that are in the grey area, defined as values between 50 and 205 (but not including 50 and 205)
mean(x > 50 & x < 205)



###KNN
install.packages('e1071', dependencies=TRUE)
library(purrr)
library(dslabs)
data("heights")
set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(heights$sex, times = 1, p = 0.5, list = FALSE)
train <- heights[-test_index,]
test <- heights[test_index,]

#1. perform knn with k values and calculate F1 scores
ks <- seq(1, 101, 3)

f_1 <- map_df(ks, function(k) {
  fit <- knn3(sex~height, data=train, k=k)
  y_hat <- predict(fit, test, type="class") #%>% 
  #factor(levels = levels(test$sex))
  #cm <- confusionMatrix(data=y_hat, reference=test$sex)
  #cm <- table(predicted=y_hat, actual=test$sex)
  f1 <- F_meas(data=y_hat, reference=factor(test$sex))
  data.frame(k=k, f1=f1)
})

max(f_1$f1)
f_1[which.max(f_1$f1),]


#2. 
library(dslabs)
library(caret)
data("tissue_gene_expression")

k = seq(1, 11, 2)

test_index <- createDataPartition(tissue_gene_expression$y, times = 1, p = 0.5, list = FALSE)
train <- tissue_gene_expression$x[-test_index,]
test <- tissue_gene_expression$x[test_index,]
train_y <- tissue_gene_expression$y[-test_index]
test_y <- tissue_gene_expression$y[test_index]

accuracy <- map_df(k, function(k) {
  fit <- knn3(x=train, y=train_y, k=k)
  y_hat <- predict(fit, data.frame(test), type="class")
  cm <- confusionMatrix(data=y_hat, reference=test_y)
  
  acc <- mean(y_hat == test_y)
  acc2 <- cm$overall["Accuracy"]
  data.frame(k=k, accuracy=acc)
})


###CROSS VALIDATION

library(tidyverse)
library(caret)

set.seed(1996, sample.kind="Rounding") 
n <- 1000
p <- 10000
x <- matrix(rnorm(n*p), n, p)
colnames(x) <- paste("x", 1:ncol(x), sep = "_")
y <- rbinom(n, 1, 0.5) %>% factor()

x_subset <- x[ ,sample(p, 100)]

#1. run cross-validation using logistic regression to fit the model
fit <- train(x_subset, y, method = "glm")
fit$results


#2. search for predictors that are most predictive of the outcome
#do this by comparing the values for the y=1 group to those in the y=2 group, for each predictor, using a t-test
#install.packages("BiocManager")
#BiocManager::install("genefilter")
library(genefilter)
tt <- colttests(x, y)

#create a vector of the p-values
pvals <- tt$p.value


#3. Create an index ind with the column numbers of the predictors that were "statistically significantly" associated with y
ind <- which(pvals < 0.01)
sum(pvals < 0.01)


#4. re-run the cross-validation after redefinining x_subset to be the subset of x defined by the columns showing "statistically significant" association with y
x_subset <- x[ ,ind]
fit <- train(x_subset, y, method = "glm")
fit$results


#5. Re-run the cross-validation again, but this time using kNN
fit <- train(x_subset, y, method = "knn", tuneGrid = data.frame(k = seq(101, 301, 25)))
ggplot(fit)


#7. Use the train() function with kNN to select the best k for predicting tissue from gene expression on the tissue_gene_expression dataset from dslabs. 
#Try k = seq(1,7,2) for tuning parameters.
data("tissue_gene_expression")
fit <- with(tissue_gene_expression, train(x, y, method = "knn", tuneGrid = data.frame( k = seq(1, 7, 2))))
ggplot(fit)



###BOOTSTRAP
library(dslabs)
library(caret)
data(mnist_27)
set.seed(1995, sample.kind="Rounding") 
indexes <- createResample(mnist_27$train$y, 10)

#1. How many times do 3, 4, and 7 appear in the first resampled index?
sum(indexes$Resample01 == 7)

#2. What is the total number of times that 3 appears in all of the resampled indexes?
x <- sapply(indexes, function(i) {
  sum(i == 3)
})
sum(x)

#3. Estimate the 75th quantile, which I know is qnorm(0.75), with the sample quantile: quantile(y, 0.75)
y <- rnorm(100, 0, 1)
qnorm(0.75)
quantile(y, 0.75)

#perform a Monte Carlo simulation with 10,000 repetitions, generating the random dataset and estimating the 75th quantile each time. 
#What is the expected value and standard error of the 75th quantile?
set.seed(1, sample.kind = "Rounding")

B <- 10^5  
M <- replicate(B, {  
  X <- rnorm(100, 0, 1)  
  quantile(X, 0.75)  
}) 

mean(M)
sd(M)


#4. In practice, I can't run a Monte Carlo simulation
#use 10 bootstrap samples to estimate the expected value and standard error of the 75th quantile

# B <- 10 
# N <- 100
# M_star <- replicate(B, {  
#   X <- sample(y, N, replace = TRUE)  
#   quantile(X, 0.75)  
# }) 
# 
# mean(M_star)
# sd(M_star)

set.seed(1, sample.kind = "Rounding")
y <- rnorm(100, 0, 1)

indexes <- createResample(y, 10)
M_star <- sapply(indexes, function(i) {
  temp_y <- y[i]
  quantile(temp_y, 0.75)
})

mean(M_star)
sd(M_star)


#5. Repeat the exercise from Q4 but with 10,000 bootstrap samples instead of 10
indexes <- createResample(y, 10000)
M_star <- sapply(indexes, function(i) {
  temp_y <- y[i]
  quantile(temp_y, 0.75)
})
mean(M_star)
sd(M_star)



###GENERATIVE MODELS

#Create a dataset of samples from just cerebellum and hippocampus, two parts of the brain, and a predictor matrix with 10 randomly selected columns using the following code:
library(dslabs)
library(caret)
library(tidyverse)
data("tissue_gene_expression")

set.seed(1993, sample.kind="Rounding") 
ind <- which(tissue_gene_expression$y %in% c("cerebellum", "hippocampus"))
y <- droplevels(tissue_gene_expression$y[ind])
x <- tissue_gene_expression$x[ind, ]
x <- x[, sample(ncol(x), 10)]


#1. estimate the accuracy of LDA
fit_lda <- train(x, y, method = "lda")
fit_lda$results["Accuracy"]

#2. Plot the mean vectors against each other and determine which predictors (genes) appear to be driving the algorithm
fit_lda$finalModel$means
t(fit_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()


#3. Repeat the exercise in Q1 with QDA
fit_qda <- train(x, y, method = "qda")
fit_qda$results["Accuracy"]

#4. 
t(fit_qda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(cerebellum, hippocampus, label = predictor_name)) +
  geom_point() +
  geom_text() +
  geom_abline()


#5. Which TWO genes drive the algorithm after performing the scaling?
fit_lda <- train(x, y, method = "lda", preProcess = "center")
fit_lda$results["Accuracy"]

t(fit_lda$finalModel$means) %>% data.frame() %>%
  mutate(predictor_name = rownames(.)) %>%
  ggplot(aes(predictor_name, hippocampus)) +
  geom_point() +
  coord_flip()

#6. Repeat the LDA analysis from Q5 but using all tissue types
library(dslabs)      
library(caret)
data("tissue_gene_expression")

set.seed(1993, sample.kind="Rounding") 
y <- tissue_gene_expression$y
x <- tissue_gene_expression$x
x <- x[, sample(ncol(x), 10)]

fit_lda <- train(x, y, method = "lda", preProcess = c("center"))
fit_lda$results["Accuracy"]



###CLASSIFICATION

#Create a simple dataset where the outcome grows 0.75 units on average for every increase in a predictor
library(rpart)
library(tidyverse)

n <- 1000
sigma <- 0.25
# set.seed(1) # if using R 3.5 or ealier
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
x <- rnorm(n, 0, 1)
y <- 0.75 * x + rnorm(n, 0, sigma)
dat <- data.frame(x = x, y = y)

#1. fit a regression tree
fit <- rpart(y ~ ., data = dat) 

#2. plot the tree
plot(fit)
text(fit)

#3. scatter plot of y versus x along with the predicted values
dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col=2)


#4. run Random Forests instead of a regression tree
library(randomForest)
fit <- randomForest(y~x, data=dat)
dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")

#5. Use the plot() function to see if the Random Forest from Q4 has converged or if I need more trees
plot(fit)

#6. It seems that the default values for the Random Forest result in an estimate that is too flexible (unsmooth). 
#Re-run the Random Forest but this time with a node size of 50 and a maximum of 25 nodes.
fit <- randomForest(y~x, data=dat, nodesize = 50, maxnodes=25)
  dat %>% 
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(x, y)) +
  geom_step(aes(x, y_hat), col = "red")
  
  
###CARET PACKAGE
library(caret)
getModelInfo("knn")
modelLookup("lm")

#1. Load the rpart package and then use the caret::train() function with method = "rpart" to fit a classification tree to the tissue_gene_expression dataset. 
#Try out cp values of seq(0, 0.1, 0.01). Plot the accuracies to report the results of the best model.
set.seed(1991, sample.kind="Rounding")
library(dslabs) 
data("tissue_gene_expression")
modelLookup("rpart")

y <- tissue_gene_expression$y
x <- tissue_gene_expression$x

train_tree <- train(x, y, method = "rpart",
                    tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)))
ggplot(train_tree, highlight = TRUE)
train_tree$bestTune


#2. Rerun the analysis in Q1, but this time with method = "rpart" and allow it to split any node by using the argument control = rpart.control(minsplit = 0)
train_tree2 <- train(x, y, method = "rpart",
                    tuneGrid = data.frame(cp = seq(0, 0.1, 0.01)), 
                    control = rpart.control(minsplit = 0))
ggplot(train_tree2, highlight = TRUE)
train_tree2$bestTune
confusionMatrix(train_tree2)

#3. Plot a tree
plot(train_tree2$finalModel)
text(train_tree2$finalModel)


#4. Use the train() function and the rf method to train a Random Forest model
modelLookup("rf")
train_rf <- train(x, y, method = "rf", 
                  nodesize = 1,
                  tuneGrid = data.frame(mtry = seq(50, 200, 25)))
ggplot(train_rf, highlight = TRUE)
train_rf$bestTune

#5. Use the function varImp() on the output of train()
imp <- varImp(train_rf)
imp

#6. Calculate the variable importance in the Random Forest call from Q4 for these seven predictors and examine where they rank
tree_terms <- as.character(unique(train_rf$finalModel$frame$var[!(train_rf$finalModel$frame$var == "<leaf>")]))
tree_terms

data_frame(term = rownames(imp$importance), 
           importance = imp$importance$Overall) %>%
  mutate(rank = rank(-importance)) %>% arrange(desc(importance)) %>%
  filter(term %in% tree_terms)
