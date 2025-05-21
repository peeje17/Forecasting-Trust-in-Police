# Forecasting-Trust-in-Police 
Forecasting Trust in Police – using Artificial Neural  Network (ANN)

## Introduction 
I will try to forecast the trust in police by using machine learning. I will be developing an Artificial Neural Network (ANN) model which will try to predict the level of trust. It is relevant for the police to know where in the country or district the trust towards police is strong or weak. This can help the police officers in the field to be aware of how to act in certain situations depending on which area they are operating in. 

The focus will be on the methods, using ANN in political science, but contact me if you want the discussion part as well.
The data used here comes from the the European social survey - you can download it here: https://ess.sikt.no/en/study/bdc7c350-1029-4cb3-9d5e-53f668b8fa74 


## Results 
Firstly I did a robust regression-model because the dependent variable is not normally distributed (Agresti, 2018, 163). Table 1 shows the multiple regression analysis, and you can observe that there are several variables which are not significant (see appendix). Rather surprising is that the variable which asks the respondent whether they feel safe when they walk alone at night (aesfdrk), is not significant. The same goes for unemployment (uemp3m), age (agea), and geography (domicil). The variable geography has a negative coefficient when going from the country to the city as expected by the theory, but it is surprising that it is insignificant. This is also confirmed by the very low adjusted r^2-value of 0,069, which indicates that the linear regression-model only explains about 7% of the variation in the data. Already now there are things that can be optimized regarding the chosen variables in order to secure a better result. 

picture

I divided the data into two parts. The first part containing 75% of the data and which means that a total of 1618 observations is used to train the ANN-model. The remaining 25% (539 observations) will be used to test whether the model can predict trust in police. Thereby, the model will be tested on unseen data, which is one of the general strengths of machine learning. For comparison I tested the regression model’s result to predict police trust. I performed a correlation test between the regression model and the unseen test data. The Pearson’s R reports a 0,258 correlation which is a weak positive correlation and means that the regression-model poorly predicts the trust in police in Germany. This is also shown in figure 1, where the regression-model only makes predictions in the range of 0,55 to 0,90 even though the actual range is between 0 and 1 (after normalising). Below are the coding for the regression model: 

```r
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
```


<div align="center">
  <img src="https://github.com/peeje17/Forecasting-Trust-in-Police/blob/main/hist_regression.png" alt="Figure 1" width="WIDTH" height="HEIGHT">
  <p>Figure 1 - Histogram of the predictions from the regression-model
</div>

The first ANN-model I tested was a model with a logistic activation function and no hidden layers. This model reported a Pearson’s R between the model and the test data of 0,255 which is worse than the regression-model. I trained several other models with different activation functions e.g. Tanh, Sigmoid, ReLU, and Softplus. The best model was an ANN-model with Softplus as activation function and with no hidden layers. The Pearson R of this model was 0.264 and thereby outperforms the regression-model by 0,06. Figure 2 depicts that the ANN-model predicts the police trust in a greater range than the regression-model, but only slightly. The only difference is that the ANN-model has a predicted range from 0,50 to 0,90 where the regression-model went from 0,55 to 0,90. Below is the code for the ANN model 

```r
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
```

<div align="center">
  <img src="https://github.com/peeje17/Forecasting-Trust-in-Police/blob/main/hist_ann-model.png" alt="Figure 1" width="WIDTH" height="HEIGHT">
  <p>Figure 2 - Histogram of the prediction from the ANN-model
</div>

The error-rate of the ANN-model is 33,06 which indicates that there is room for improvement. The number of steps done in the network was more than 5000, which tells us something about the complexity of the network. I also tried to improve the model by adding more layers and more nodes in each layer. This only made the prediction worse, which is probably caused by the model overfitting the training data making it unable to predict the test data. This is still surprising because just by adding 1 hidden layer with two nodes, the result decreased to a Pearson R of 0,245. The model with 2 hidden layers did more than 139.775 steps which then resulted in overfitting. I tried including more hidden layers and nodes, but this continued to make the prediction worse.  

Another approach to improve the ANN-model is to do a cross validation. Normally the cross validation is done with ten folds and thereby using the dataset in different combinations. The cross validation approach takes the first fold and includes 90% of the data and builds a model, and thereafter tests the result on the remaining 10% (Lantz, 2019, 341). The cross validation does this ten times, and the average is then calculated. When doing this, the model does not begin at a random, but all combinations of test and training data are tried. The result from the cross validation can then be compared by measuring the error rate, which determines whether the cross validation optimized the model (Alice, 2015). Cross validation can also be included to protect against overfitting which was a concern with my model (Wordliczek, 2023, 48). Cross validation is also a computationally intensive method, and that is why this step was not possible because of lack of computational power. So, cross validation could have improved the ANN-model by protecting the model from overfitting data. 


## Conclusion 
The ANN-model did not perform as expected and cannot (yet) be used by the police in practice. The ANN-model only slightly outperformed the multiple regression analysis, which was surprising because comparisons between ANN and multiple regression models in other research of political science show that ANN is the better performing method of the two (Wordliczek, 2023). The analysis also showed that there were several improvements that can be implemented to the model. One of them was to reconsider the chosen variables and maybe also the dataset. Another tool to consider was cross validation which future research can implement to the ANN-model and thereby get a better result. Future research should still try to improve the model in terms of predicting police trust, because in that way the police can improve its relationship to the citizens, which is one of the most important tasks for a democratic state. 


## Bibliography 

Agresti, A. (2018). Statistical methods for the social sciences (Fifth, Global). Pearson. https://go.exlibris.link/SZRwm4VF

Alemika, E. E. O. (1999). Police-Community Relations in Nigeria: What Went Wrong? Policing a Democracy, Lagos: Centre for Law Enforcement Education.

Alice, M. (2015, September 23). Fitting a neural network in R; neuralnet package | R-bloggers. https://www.r-bloggers.com/2015/09/fitting-a-neural-network-in-r-neuralnet-package/

Almond, G. A., & Verba, S. (1989). The civic culture: Political attitudes and democracy in five nations (New). Sage Publications. https://go.exlibris.link/xBzkPV3G

European Social Survey ERIC (ESS ERIC). (2019). ESS Round 9: European Social Survey Round 9 Data [Data set]. Sikt - Norwegian Agency for Shared Services in Education and Research. https://doi.org/10.21338/NSD-ESS9-2018

Glaeser, E. L., & Sacerdote, B. (1999). Why is There More Crime in Cities? Journal of Political Economy, 107(S6), S225–S258. https://doi.org/10.1086/250109

Goldsmith, A. (2005). Police reform and the problem of trust. Theoretical Criminology, 9(4), 443–470. https://doi.org/10.1177/1362480605057727

Lantz, B. (2019). Machine learning with R: Expert techniques for predictive modeling. Packt publishing ltd.

Warren, M. (1999). Democracy and trust. Cambridge University Press. https://doi.org/10.1017/CBO9780511659959

Weber, P., Weber, N., Goesele, M., & Kabst, R. (2018). Prospect for knowledge in survey data: An artificial neural network sensitivity analysis. Social Science Computer Review, 36(5), 575–590.

Wordliczek, Ł. (2023). Neural Networks and Political Science: Testing the Methodological Frontiers. Empiria (Madrid), 57, 37–62. https://doi.org/10.5944/empiria.57.2023.36429




