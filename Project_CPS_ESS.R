library(dplyr)
library(tidyverse)
library(haven)
library(caret)
library(labelled)
library(cowplot)
library(lattice)
library(Cubist)
library(neuralnet)
library(car)
library(robustbase)


                                  ### Step 1: Read ESS Data ###
ess <- read.csv("ESS9e03_1.csv")
# The dependent variable, which I want to predict 
hist(ess$trstplc)
# trstplc: please tell me on a score of 0-10 how much you personally trust the Police. 
# 0 means you do not trust an institution at all, and 10 means you have complete trust

                                  ### Step 2 Exploring and preparing data ### 
# I only want it from Germany 
ess <- filter(ess, cntry == "DE")


# I select the variables from the dataset
ess = ess %>% select("eduyrs", "uemp3m", "gndr", "agea", "dscrgrp", "domicil", 
                     "emplrel", "aesfdrk", "rlgblg", "ppltrst", "sclact", "trstplc")

# Recode variables
ess$sclact_new = recode(ess$sclact, "8 = NA")
# it's a ordinal sclae - see how it goes otherwise to an nominal
table(ess$sclact) # 10 missings
table(ess$sclact_new)
hist(ess$sclact)


ess$aesfdrk_new = recode(ess$aesfdrk, "1:2 = 0 ; 3:4 = 1 ; 8 = NA")
# 0 = safe or very safe and 1 = unsafe or very unsafe 
table(ess$aesfdrk)
table(ess$aesfdrk_new)

ess$gndr_new = recode(ess$gndr, "2 = 0")
# 0 = female and male = 1
table(ess$gndr)
table(ess$gndr_new)                   

ess$eduyrs_new = recode(ess$eduyrs, "77 = NA ; 88 = NA")
table(ess$eduyrs)
table(ess$eduyrs_new)

ess$uemp3m_new = recode(ess$uemp3m, "8 = NA ; 2 = 0 ; 1 = 1")
#0 = no and 1 = yes
table(ess$uemp3m)
table(ess$uemp3m_new)


ess$emplrel_new = recode(ess$emplrel, "6 = NA ; 7 = NA ; 8 = NA ; 3 = NA ; 1 = 0 ; 2 = 1")
# employee = 0 and self-employed = 1
table(ess$emplrel) # 134+35 NAs 
table(ess$emplrel_new)
 
ess$agea_new = recode(ess$agea, "999 = NA")
table(ess$agea)
table(ess$agea_new)
hist(ess$agea_new)

ess$rlgblg_new = recode(ess$rlgblg, "7 = NA ; 8 = NA ; 1 = 1 ; 2 = 0")
#Do you consider yourself as belonging to any particular religion or denomination?
# 0=no and 1 = yes
table(ess$rlgblg)
table(ess$rlgblg_new)


ess$ppltrst_new = recode(ess$ppltrst, "")
table(ess$ppltrst)
hist(ess$ppltrst)
table(ess$ppltrst_new)


ess$dscrgrp_new = recode(ess$dscrgrp, "7 = NA ; 8 = NA ; 2 = 0")
# yes=1 and no=o
table(ess$dscrgrp) # 8 NAs
table(ess$dscrgrp_new)

ess$domicil_new = recode(ess$domicil, "1:2 = 1 ; 3:5 = 0")
table(ess$domicil)
table(ess$domicil_new)


ess$trstplc_new = recode(ess$trstplc, "88 = NA")
table(ess$trstplc)
table(ess$trstplc_new)
hist(ess$trstplc_new)

# I remove the old variables 
ess <- select(ess, -eduyrs, -uemp3m, -gndr, -rlgblg, -agea, -dscrgrp, -domicil,
              -emplrel, -ppltrst, -aesfdrk, -sclact, -trstplc)


# I use lmrob, because the independent variable is not normal distributed 

lm = lmrob(trstplc_new ~ ., data = ess)
summary(lm)


# I remove the NA's pairwise 
#ess_ona2 <- drop_na(ess)
ess_ona = na.omit(ess, pairwise = TRUE) # it removes 201 observations dataset - 2157 observations left


# Normalize the data

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
ess_norm <- as.data.frame(lapply(ess_ona, normalize))
summary(ess_norm$trstplc_new)
mean(ess_norm$trstplc_new)


# LM test 3: to see wheter the results are better with normalized results 
lm2 = lmrob(trstplc_new ~ ., data = ess_norm)
summary(lm2) # no difference

str(ess_norm) # all variables are now numeric values 

# Descriptive 
library(stargazer)
des = summary(ess_norm)
des
stargazer(ess_norm)
stargazer(lm2)
hist(ess_norm$trstplc_new,
     main = "Histogram of police trust (dependent variable)",
     xlab = "trust in the police")



#Dividing the data in training and test data
# training data 75% of the data and test data 25% 
ess_train <- ess_norm[1:1618, ]
mean(ess_train$trstplc_new)
# evaluation data 25% of the data
ess_test <- ess_norm[1619:2157, ]
mean(ess_test$trstplc_new)



                      ### Step 3 - Training a model on the data ### 
# model 1 
library(neuralnet)
set.seed(12345)
ess_model1 <- neuralnet(trstplc_new ~ ., data = ess_train)
ess_model1


plot(ess_model1) # 

                      ### Step 4 - evaluating model performance ### 

model_results1 <- compute(ess_model1, ess_test[-12])
predicted1 <- model_results1$net.result
cor(predicted1, ess_test$trstplc_new) # 0.255


                            ### Step 5: Improving the model performance 

# Model 2 with 2 hidden layers
set.seed(12345)
ess_model2 <- neuralnet(trstplc_new ~ ., data = ess_train, hidden = 2, stepmax = 1e+06)
ess_model2
plot(ess_model2) #  


model_results2 <- compute(ess_model2, ess_test[-12])
predicted_2 <- model_results2$net.result
cor(predicted_2, ess_test$trstplc_new) # 0.245 

# Model 3 tanh
set.seed(12345)
ess_model3 <- neuralnet(trstplc_new ~ ., data = ess_train, act.fct = tanh)

plot(ess_model3)# 


model_results3 <- compute(ess_model3, ess_test[1:11])
predicted_ess3 <- model_results3$net.result
cor(predicted_ess3, ess_test$trstplc_new) # 0.254 



## Model 6 ## - with softplus - the model which is presented in the paper
softplus <- function(x) log(1 + exp(x))
set.seed(12345)
ess_model6 <- neuralnet(trstplc_new ~ ., 
                        data = ess_train, act.fct = softplus, stepmax = 1e+06)
ess_model6

plot(ess_model6) # 


model_results6 <- compute(ess_model6, ess_test[1:11])
predicted_ess6 <- model_results6$net.result
cor(predicted_ess6, ess_test$trstplc_new) # 0.264
# further exploration
plot(predicted_ess6, ess_test$trstplc_new)
hist(predicted_ess6,
     main = "Predicted values for the ANN-model",
     xlab = "predicted police trust") # it predicts in a lesser frame) #it predicts in a better way than lm 
hist(ess_test$trstplc_new)

# Model 7 with 2 hidden layers and softplus   
softplus <- function(x) log(1 + exp(x))
set.seed(12345)
ess_model7 <- neuralnet(trstplc_new ~ ., data = ess_train, hidden = 2, act.fct = softplus, stepmax = 1e+06)

plot(ess_model7) #
ess_model7


model_results7 <- compute(ess_model7, ess_test[1:11])
predicted_ess7 <- model_results7$net.result
cor(predicted_ess7, ess_test$trstplc_new) # 0.229



# Model 8 with 4,2 hidden layers and softplus 
softplus <- function(x) log(1 + exp(x))
set.seed(12345)
ess_model8 <- neuralnet(trstplc_new ~ ., data = ess_train, hidden = c(4,2), act.fct = softplus, stepmax = 1e+06)

plot(ess_model8)
ess_model8


model_results8 <- compute(ess_model8, ess_test[1:11])
predicted_ess8 <- model_results8$net.result
cor(predicted_ess8, ess_test$trstplc_new) # 0.029 



# Model 11 - 4,2 
set.seed(12345)
ess_model11 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, hidden = c(4,2))


plot(ess_model11)#  


model_results11 <- compute(ess_model11, ess_test[1:11])
predicted11 <- model_results11$net.result
cor(predicted11, ess_test$trstplc_new) # 0.231 



# Model 15 more hidden layers (8-4-2) 
set.seed(12345)
ess_model15 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, hidden = c(8,4,2), stepmax = 1e+06)


plot(ess_model15) # 


model_results15 <- compute(ess_model15, ess_test[1:11])
predicted15 <- model_results15$net.result
cor(predicted15, ess_test$trstplc_new) # 0.2187  


#Model 17: sigmoid activation  
sigmoid = function(x) 1 / (1 + exp(-x))
set.seed(12345)
ess_model17 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, act.fct = sigmoid, stepmax = 1e+06)

plot(ess_model17) # 


model_results17 <- compute(ess_model17, ess_test[1:11])
predicted_ess17 <- model_results17$net.result
cor(predicted_ess17, ess_test$trstplc_new) # 0.255 


# model 18 4,2 and tanh
set.seed(12345)
ess_model18 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, hidden = c(4, 2), act.fct = tanh, stepmax = 1e+06)

plot(ess_model18) # 


model_results18 <- compute(ess_model18, ess_test[1:11])
predicted_ess18 <- model_results18$net.result
cor(predicted_ess18, ess_test$trstplc_new) # took more than 10min  

 


# Model 22 - sigmoid and 4,2
sigmoid = function(x) 1 / (1 + exp(-x))
set.seed(12345)
ess_model22 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, hidden = c(4,2), act.fct = sigmoid, stepmax = 1e+06)

plot(ess_model22) #  


model_results22 <- compute(ess_model22, ess_test[1:11])
predicted_ess22 <- model_results22$net.result
cor(predicted_ess22, ess_test$trstplc_new) # 0.234   


# Model 23 - ReLU
relu <- function(x) sapply(x, function(z) max(0,z))
ess_model23 <- neuralnet(trstplc_new ~ .,
                         data = ess_train, act.fct = relu, stepmax = 1e+06)

plot(ess_model23) #  


model_results23 <- compute(ess_model23, ess_test[1:11])
predicted_ess23 <- model_results23$net.result
cor(predicted_ess23, ess_test$trstplc_new) # 0.116   



# In comparation with multiple regression model 
lm_last = lmrob(trstplc_new ~ ., data = ess_train)

summary(lm_last)

model_results_lm <- predict(lm_last, ess_test[1:11])
predicted_ess_lm <- model_results_lm
cor(predicted_ess_lm, ess_test$trstplc_new) # 0.258
# further exploration
plot(predicted_ess_lm, ess_test$trstplc_new)
hist(predicted_ess_lm,
     main = "Predicted values for the regression-model",
     xlab = "predicted police trust") # it predicts in a lesser frame


# Cross validation - could not be accomplished

folds = createFolds(ess_norm$trstplc_new, k = 10)
str(folds)

# now divided each fold with 90% training data and 10% test data
ess01_test <- ess_norm[folds$Fold01, ]
ess01_train <- ess_norm[-folds$Fold01,] 

# Or faster

ess_results <- lapply(folds, function(x) {
  ess_train_folds <- ess_norm[-x, ]
  ess_test_folds <- ess_norm[x, ]
  ess_model_folds <- train(trstplc_new ~ ., data = ess_train_folds, method = "neuralnet")
  ess_pred_folds <- predict(ess_model_folds, ess_test_folds)
  ess_actual_folds <- ess_test_folds$trstplc_new
  layer <- layer(data.frame(ess_actual_folds, ess_pred_folds))$value
  return(layer)
})
# does not finish, because it takes to long
