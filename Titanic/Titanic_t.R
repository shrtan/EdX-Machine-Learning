library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)


#1. Training and Test sets
#Split titanic_clean into test and training sets - after running the setup code, it should have 891 rows and 9 variables.
#Set the seed to 42, then use the caret package to create a 20% data partition based on the Survived column. Assign the 20% partition to test_set and the remaining 80% partition to train_set.
set.seed(42, sample.kind = "Rounding")
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE)
test <- titanic_clean[test_index,]
train <- titanic_clean[-test_index,]

#What proprotion in the training set survived?
mean(train$Survived == 1)


#2. Baseline prediction by guessing the outcome
#The simplest prediction method is randomly guessing the outcome without using additional predictors
set.seed(3, sample.kind = "Rounding")
guess_pred <- sample(c(0, 1), size = nrow(test), replace = TRUE)
#What is the accuracy of this guessing method?
mean(guess_pred == test$Survived)


#3a. Predicting survival by sex
(train %>% filter(Sex == "male", Survived == 1) %>% nrow())/length(train$Sex[train$Sex == "male"]) 
(train %>% filter(Sex == "female", Survived == 1) %>% nrow())/length(train$Sex[train$Sex == "female"])

train %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1))

#3b. Predicting survival by sex
test %>%
  summarize((sum(Sex == 'female' & Survived == 1) 
             + sum(Sex == 'male' & Survived == 0)) / n())


#4a. In the training set, which class(es) (Pclass) were passengers more likely to survive than die?
p_class_survival <- train %>% group_by(Pclass) %>% 
  summarise(Survived_percent = mean(Survived == 1)) %>% 
  mutate(Suvived = ifelse(Survived_percent > 0.5, 1, 0))

#4b. Predict survival using passenger class on the test set: predict survival if the survival rate for a class is over 0.5, otherwise predict death.
pclass_predict <- lapply(test$Pclass, function(pclass) {
  if(pclass == p_class_survival$Pclass[1]) p_class_survival$Suvived[1]
  if(pclass == p_class_survival$Pclass[2]) p_class_survival$Suvived[2]
  if(pclass == p_class_survival$Pclass[3]) p_class_survival$Suvived[3]
})

pclass_predict = NULL
i = 1
for (pc in test$Pclass) {
  if(pc == p_class_survival$Pclass[1]) {pclass_predict[i] <- p_class_survival$Suvived[1]}
  if(pc == p_class_survival$Pclass[2]) {pclass_predict[i] <- p_class_survival$Suvived[2]}
  if(pc == p_class_survival$Pclass[3]) {pclass_predict[i] <- p_class_survival$Suvived[3]}
  i = i+1
}

#What is the accuracy of this guessing method?
mean(pclass_predict == test$Survived)


#4c. Use the training set to group passengers by both sex and passenger class.
#Which sex and class combinations were more likely to survive than die (i.e. >50% survival)?
train %>% group_by(Sex, Pclass) %>% summarise(Survived_percent = sum(Survived == 1))

survival_class <- titanic_clean %>%
  group_by(Sex, Pclass) %>%
  summarize(PredictingSurvival = ifelse(mean(Survived == 1) > 0.5, 1, 0))
survival_class

condition = (test$Sex == "female" & test$Pclass == 1) | (test$Sex == "female" & test$Pclass == 2)
gender_pc <- ifelse(condition, 1, 0)

#accuracy of thisprediction
mean(gender_pc == test$Survived)


#5. Create confusion matrices for the sex model, class model, and combined sex and class model
library(broom)

# Confusion Matrix: sex model
sex_model <- train %>%
  group_by(Sex) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))

test_set1 <- test %>%
  inner_join(sex_model, by = 'Sex')

cm1 <- confusionMatrix(data = factor(test_set1$Survived_predict), reference = factor(test_set1$Survived))
cm1 %>% tidy() %>% 
  filter(term == 'sensitivity' | term == 'specificity' | term == 'balanced_accuracy') 

# Confusion Matrix: Pclass model
pclass_model <- train %>%
  group_by(Pclass) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))

test_set2 <- test %>%
  inner_join(pclass_model, by = 'Pclass')

cm2 <- confusionMatrix(data = factor(test_set2$Survived_predict), reference = factor(test_set2$Survived))
cm2 %>% tidy() %>% 
  filter(term == 'sensitivity' | term == 'specificity' | term == 'balanced_accuracy') 

# Confusion Matrix: Pclass and Sex combined model
combined_model <- train %>%
  group_by(Pclass, Sex) %>%
  summarize(Survived_predict = ifelse(mean(Survived == 1) > 0.5, 1, 0))

test_set3 <- test %>%
  inner_join(combined_model, by = c('Pclass', 'Sex'))

cm3 <- confusionMatrix(data = factor(test_set3$Survived_predict), reference = factor(test_set3$Survived))
cm3 %>% tidy() %>% 
  filter(term == 'sensitivity' | term == 'specificity' | term == 'balanced_accuracy') 


#6. Calculate  scores for the sex model, class model, and combined sex and class model
F_meas(data = factor(test_set1$Survived_predict), reference = factor(test_set1$Survived))
F_meas(data = factor(test_set2$Survived_predict), reference = factor(test_set2$Survived))
F_meas(data = factor(test_set3$Survived_predict), reference = factor(test_set3$Survived))


#7. Survival by fare - LDA and QDA
set.seed(1, sample.kind = "Rounding")
#lda model
fit_lda <- train(Survived ~ Fare, data=train, method = 'lda')
survived_hat <- predict(fit_lda, test)
mean(survived_hat == test$Survived)

#qda model
fit_qda <- train(Survived ~ Fare, data=train, method = 'qda')
survived_hat <- predict(fit_qda, test)
mean(survived_hat == test$Survived)


#8. Logistic regression models
#Train a logistic regression model using age as the only predictor
set.seed(1, sample.kind = "Rounding")
fit_sex <- glm(Survived ~ Sex, data=train, family = binomial)
survived_hat <- predict(fit_sex, test)
survived_hat <- ifelse(survived_hat >= 0, 1, 0)
mean(survived_hat == test$Survived)

#Train a logistic regression model using four predictors: sex, class, fare, and age
fit_4p <- glm(Survived ~ Sex + Pclass + Age + Fare, data=train, family = binomial)
survived_hat <- predict(fit_4p, test, type = "response")
survived_hat <- ifelse(survived_hat >= 0.5, 1, 0)
mean(survived_hat == test$Survived)

#Train a logistic regression model using all predictors
fit_all <- glm(Survived ~ ., data=train, family = binomial)
survived_hat <- predict(fit_all, test, type = "response")
survived_hat <- ifelse(survived_hat >= 0.5, 1, 0)
mean(survived_hat == test$Survived)


#9. kNN model
#a. Train a kNN model. Try tuning with k = seq(3, 51, 2).
#What is the optimal value of the number of neighbors k?
set.seed(6, sample.kind = "Rounding")
k <- seq(3, 51, 2)
fit_knn <- train(Survived ~ ., data = train, method = "knn", 
                 tuneGrid = data.frame(k))
fit_knn$bestTune

#plotting k values
ggplot(fit_knn)

#b. Of 7, 11, 17 and 21, which yields the highest accuracy
fit_knn$results %>% filter(k %in% c(7, 11, 17, 21)) %>% pull(Accuracy)

#c. What is the accuracy of the kNN model on the test set?
survived_hat <- predict(fit_knn, test)
mean(survived_hat == test$Survived)


#10. Use 10-fold cross-validation where each partition consists of 10% of the total. 
#Try tuning with k = seq(3, 51, 2).
set.seed(8, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9) 

train_knn_cv <- train(Survived ~ ., method = "knn",  
                      data = train, 
                      tuneGrid = data.frame(k = seq(3, 51, 2)), 
                      trControl = control) 
#optimal value of k
train_knn_cv$bestTune

#accuracy of test set
survived_hat <- predict(train_knn_cv, test)
mean(survived_hat == test$Survived)


#11a. Train a decision tree with the rpart method. Tune the complexity parameter with cp = seq(0, 0.05, 0.002).
set.seed(10, sample.kind = "Rounding")
fit_tree <- train(Survived ~ ., method = "rpart",  
                  data = train, 
                  tuneGrid = data.frame(cp = seq(0, 0.05, 0.002))) 
fit_tree$bestTune

fit_tree$results %>% filter(cp == 0.028)

#accuracy with test set
survived_hat <- predict(fit_tree, test)
mean(survived_hat == test$Survived)

#11b. Inspect the final model and plot the decision tree.
#Which variables are used in the decision tree?
rpart.plot(fit_tree$finalModel) 
rpart.plot(fit_tree$finalModel, type = 3, box.palette = c("red", "green"), fallen.leaves = TRUE) 


#12. Random forest model
set.seed(14, sample.kind = "Rounding")

fit_rf <- train(Survived ~ ., data = train, method = "rf",
                tuneGrid = data.frame(mtry = seq(1:7)), ntree = 100)
fit_rf$bestTune

survived_hat <- predict(fit_rf, test)
confusionMatrix(data = factor(survived_hat), reference = factor(test$Survived))
mean(survived_hat == test$Survived)
varImp(fit_rf$finalModel)