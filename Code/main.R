# -----------------------------------------------------------------------------
# Load Data & Libraries
# -----------------------------------------------------------------------------

setwd("C://Users/Aritz/Desktop/1.2/Seminar of Applied Statistics/HOMEWORK")

library(GGally) # ggpairs
library(rpart) # tree
library(rpart.plot) # tree visualization
library(caret) # confusion matrix, data partition, cross validation, etc.
source("draw_confusion_matrix.R") # confusion matrix visualization
library(nnet) # NN
library(NeuralNetTools) # NN visualization
library(neuralnet) # also NN
library(randomForest) # random forest
library(MASS) # LDA
library(e1071) # SVM
library(naivebayes) # Naive Bayes

credit =read.csv("GermanCredit.csv", dec=".", header=T, sep=";") 

# -----------------------------------------------------------------------------
# Data Cleaning
# -----------------------------------------------------------------------------

# str(credit)

# Correct nature of data (numerical/categorical(/binary))
# Categorical/binary variables
cols = c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE", "RADIO.TV", "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT", "MALE_DIV", "MALE_SINGLE", "MALE_MAR_or_WID", "CO.APPLICANT", "GUARANTOR", "PRESENT_RESIDENT", "REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT", "OWN_RES", "NUM_CREDITS", "JOB", "TELEPHONE", "FOREIGN", "RESPONSE")
credit[cols] = lapply(credit[cols], factor)  
# Numerical variables that can be treated as categorical
cols = c("INSTALL_RATE", "NUM_DEPENDENTS")
credit[cols] = lapply(credit[cols], factor) 

summary(credit)
# We have 300 '0' and 700 '1'
# This might affect our ability to predict
# Models might be able to predict better '1' than '0' simply because we have more information from '1'

# We don't need OBS.
credit = credit[,-1]

# We find observations with invalid inputs. 
# Since it is not possible to 'guess' what the correct inputs are, we remove them.
# Education -1 (has to be 0/1)
remove = which(credit$EDUCATION==-1)
credit = credit[-remove,]
credit$EDUCATION = droplevels(credit$EDUCATION) # remove factor level
# Guarantor 2 (has to be 0/1)
remove = which(credit$GUARANTOR==2)
credit = credit[-remove,]
credit$GUARANTOR = droplevels(credit$GUARANTOR) # remove factor level

# Clear outlier (probably wrong input)
# Age 125
# Modify as median of age (another option would have been removing the observation)
modify = which(credit$AGE==125)
credit[modify,]$AGE = median(credit$AGE)

# Transformations/new variables
# NOT USED IN THE END BECAUSE THEY ARE NOT FOUND TO BE BENEFICIAL

# Created new variable (new/used/no car) from 2 variables (New car, Used car)
# New car = 2
# Used car = 1
# No car = 0
# CAR = ifelse(credit$NEW_CAR==1, 2, ifelse(credit$USED_CAR==1, 1, 0))
# credit['CAR'] = as.factor(CAR)

# Gender: female(0)/male(1)
# GENDER = ifelse(credit$MALE_DIV==0 & credit$MALE_SINGLE==0 & credit$MALE_MAR_or_WID==0, '0', '1')
# credit['GENDER'] = as.factor(GENDER)

# -----------------------------------------------------------------------------
# Exploratory data analysis
# -----------------------------------------------------------------------------

# We want to see differences between response 0 and 1

# Numerical data
nums = unlist(lapply(credit, is.numeric))  
nums['RESPONSE']=TRUE # to keep response
credit_num = credit[ , nums]
# Explore
ggpairs(credit_num, ggplot2::aes(colour=RESPONSE))
# Slight differences between response types

# Categorical (/binary) data
nums = unlist(lapply(credit, is.factor))  
credit_cat = credit[ , nums]

# Function to plot side-by-side barplots
pink = "#F8766D"
blue = "#00BFC4"
my_plot = function(variable_name, data){
  if (variable_name!='RESPONSE'){
    # Contingency table
    table = table(credit$RESPONSE, data, dnn=c("Response", variable_name))
    # Percentage table
    table_percentage = apply(table, 2, function(x){x*100/sum(x,na.rm=T)})
    # Barplot (stacked)
    barplot(table_percentage , border="white", main=variable_name, col=c(pink, blue ), ylab=expression(bold('Percentage %')))
    # Add mean lines
    # Mean used because median can be misleading, since we have categorical variables with not many levels 
    mean0 = mean(as.numeric(data[credit$RESPONSE==0]))
    mean1 = mean(as.numeric(data[credit$RESPONSE==1]))
    segments(x0=mean0,y0=0,x1=mean0,y1=100,col='black', lwd=2, lty=2) # dashed line -> response = 0 (bad credit)
    segments(x0=mean1,y0=0,x1=mean1,y1=100,col='black', lwd=2, lty=1) # normal line -> response = 1 (good credit)
    
    # Boxplot (not really usefull for categorical variables)
    # boxplot(as.numeric(data)-1~credit$RESPONSE, col=c("red", "green"), horizontal = TRUE)
  }
}

# Plot for CHK_ACCT
variable_name = "Checking account status"
data = credit_cat[,1]
my_plot(variable_name, data)
legend("topright", title=as.expression(bquote(bold("Applicant's credit risk"))), legend = c("Bad", "Good"), lty=c(1,1), lwd=c(5,5), col=c(pink, blue))

# Plot all categorical variables
par(mfrow=c(3,3), mai = c(0.25, 0.25, 0.25, 0.25))
for (i in 1:ncol(credit_cat)){
  variable_name = names(credit_cat)[i]
  data = credit_cat[,i]
  my_plot(variable_name, data)
  # if (i%%9==0){ # add legend after a multiple of 9 plots
  #   legend(x = "top", legend = c("0", "1"), col=c("red", "green"))
  # }
}
# We look for significant difference on means & difference on percentages for each response type
# Clear guesses: CHK_ACCT, HISTORY, SAV_ACCT
# Not so clear guesses: NEW_CAR, USED_CAR, RADIO.TV, EMPLOYMENT, INSTALL_RATE, MALE_SINGLE, REAL_ESTATE, PROP_UNKN_NONE, OTHER_INSTALL, RENT, OWN_RES

# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

# 0. Train/test split
# 1. Logistic regression
# 2. Neural network
# 3. Tree
# 4. Random forest
# 5. LDA
# 6. KNN
# 7. SVM
# 8. Naive Bayes

# -----------------------------------------------------------------------------
# 0. TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------

# Take only certain variables - OPTIONAL
# meaningful = c("CHK_ACCT","DURATION","SAV_ACCT","HISTORY","AMOUNT","RESPONSE")
# credit = credit[meaningful]

# Creating indices
set.seed(123)
trainIndex = createDataPartition(credit$RESPONSE,p=0.75,list=FALSE)
# Splitting data into training/testing data
credit.train = credit[trainIndex,] #training data (75% of data)
credit.test = credit[-trainIndex,] #testing data (25% of data)

# Prediction threshold for predicting clients as 'good'
# The prediction will be 'good' if the prediction prob. for 'good credit' is > threshold
# Conventional model: threshold = 0.5
# Conservative model (do not grant credits so easily): threshold > 0.5
threshold = 0.75

# -----------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# -----------------------------------------------------------------------------

fit_glm = glm(RESPONSE~.,data=credit.train,family = "binomial")
varImp(fit_glm) # importance
glm.pred = predict(fit_glm, newdata=credit.test, type="response")
glm.pred = ifelse(glm.pred<threshold,0,1)
glm.pred = as.factor(glm.pred)
# Confusion matrix
cm = confusionMatrix(data = glm.pred, reference = credit.test$RESPONSE)
# Visualize
draw_confusion_matrix(cm,'LOGISTIC REGRESSION')

# -----------------------------------------------------------------------------
# 2. NEURAL NETWORK
# -----------------------------------------------------------------------------

# nnet with caret (cross validation to choose size and decay)
nnetFit = caret::train(RESPONSE ~ ., data = credit.train, method = "nnet", preProcess = "range", trace = FALSE, trControl = trainControl(method = "cv"))
nnetFit # takes size=1 and decay=0.1
# Visualize
par(mar = numeric(4))
plotnet(nnetFit, pos_col = "darkgreen", neg_col = "darkblue")
# Impossible to interpret
# Predict
nn.pred = predict(nnetFit, credit.test, type = "prob")
nn.pred = ifelse(nn.pred[,2]<threshold,0,1)
nn.pred = as.factor(nn.pred)
cm = confusionMatrix(data = nn.pred, reference = credit.test$RESPONSE)
par(pty='m')
draw_confusion_matrix(cm,'NEURAL NETWORK')

# -----------------------------------------------------------------------------
# 3. TREE
# -----------------------------------------------------------------------------

## rpart: Recursive Partitioning and Regression Trees
credit.tree = rpart(RESPONSE ~ ., method = "class", data = credit.train, control = rpart.control(minsplit = 4, cp = 1e-05), model = TRUE)
# credit.tree$splits # To see splits (but this tree is too complex)
var_importance = credit.tree$variable.importance
# Main ones: AMOUNT, CHK_ACCT, DURATION, AGE, SAV_ACCT, HISTORY
# But this only represents how many times each variable is used to split the tree
# If it's near the nodes, it's not so important
# Visualize the complex tree (hard to interpret)
par(mfrow=c(1,1), pty = "s", mar = c(1, 1, 1, 1))
plot(credit.tree, cex = 1)
text(credit.tree, cex = 0.6)

# The Complexity Table
# printcp(credit.tree)
# Plot error, CV error, etc.
par(pty = "s", mar=c(3,3,1,1))
plotcp(credit.tree, ylim=c(0.6,1.1))
with(credit.tree, {
  lines(cptable[, 2] + 1, cptable[, 3], type = "b", col = "red")
  legend(11, 0.8, c("Train Error", "CV Error", "min(CV Error)+1SE"), 
         lty = c(1, 1, 2), col = c("red", "black", "black"), bty = "n", cex = 1)
})

# We prune the tree at the simples tree (lowest level)
cp = credit.tree$cptable
opt = which.min(credit.tree$cptable[, "xerror"])
r = cp[, 4][opt] + cp[, 5][opt]
rmin = min(seq(1:dim(cp)[1])[cp[, 4] < r])
cp0 = cp[rmin, 1]
cat("Number of nodes of the pruned tree: ", cp[rmin, 2]+1, "\n")

credit.prune = prune(credit.tree, cp = 1.01*cp0)
# Plot the pruned tree
par(mfrow=c(1,1), pty = "s", mar = c(1, 1, 1, 1))
plot(credit.prune, cex = 1)
text(credit.prune, cex = 0.6)
# Another visualization (clearer)
prp(credit.prune)
# We see that main ones are CHK_ACCT, DURATION, SAV_ACCT, HISTORY

# Predict
tree.pred = predict(credit.prune, credit.test, type = "prob")
tree.pred = ifelse(tree.pred[,2]<threshold,0,1)
tree.pred = as.factor(tree.pred)
cm = confusionMatrix(data = tree.pred, reference = credit.test$RESPONSE)
par(pty='m')
draw_confusion_matrix(cm, 'TREE')

# -----------------------------------------------------------------------------
# 4. RANDOM FOREST
# -----------------------------------------------------------------------------

form = formula(RESPONSE ~ .)
credit.RF = randomForest(formula=form,
                          data=credit.train,
                          ntree=500,
                          importance=TRUE,
                          localImp=TRUE,
                          na.action=na.roughfix,
                          replace=FALSE,
                          nodesize=10)

# Importance of variables
varImp(credit.RF, scale = TRUE)[order(varImp(credit.RF, scale = TRUE)),]
# Main ones: CHK_ACCT, DURATION, HISTORY, SAV_ACCT, AMOUNT

# Improvement with more trees and then stabilizes
plot(credit.RF, main="", col=c('black',pink,blue), lwd=2)
legend(x=170,y=0.45, legend = c("Out-of-bag samples", "Bad credit", "Good credit"), lty=c(1,1,1), lwd=c(5,5,5), col=c('black',pink,blue), cex=1.3)
text("Number of trees", x=250,y=0.06, cex=1.3)
# Test set predictions
rf.pred = predict(credit.RF, credit.test, type = "prob")
rf.pred = ifelse(rf.pred[,2]<threshold,0,1)
rf.pred = as.factor(rf.pred)
# Confusion matrix
cm = confusionMatrix(data = rf.pred, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'RANDOM FOREST')

# BOOSTING GBM 
# NOT USED IN THE END
# 10-fold cross validation
# fitControl = trainControl(method = "cv",
#                            number = 10)
# tune_Grid =  expand.grid(interaction.depth = 2,
#                           n.trees = 500,
#                           shrinkage = 0.1,
#                           n.minobsinnode = 10)
# fit = train(RESPONSE ~ ., data = credit.train,
#              method = "gbm",
#              trControl = fitControl,
#              verbose = FALSE,
#              tuneGrid = tune_Grid)
# 
# # Test set
# pred = predict(fit, credit.test, type = "prob")
# pred = ifelse(pred[,2]<threshold,0,1)
# pred = as.factor(pred)
# # Confusion matrix
# cm = confusionMatrix(data = pred, reference = credit.test$RESPONSE)
# # Visualize
# draw_confusion_matrix(cm,'GBM')

# -----------------------------------------------------------------------------
# 5. LDA
# -----------------------------------------------------------------------------

# Estimate preprocessing parameters
preproc.param = credit.train %>% 
  preProcess(method = c("center", "scale"))
# Transform the data using the estimated parameters
train.transformed = preproc.param %>% predict(credit.train)
test.transformed = preproc.param %>% predict(credit.test)

# Fit the model
model = lda(RESPONSE~., data = train.transformed)
# Make predictions
predictions = model %>% predict(test.transformed)
predictions = ifelse(predictions$posterior[,2]<threshold,0,1)
predictions = as.factor(predictions)
# Confusion matrix
cm = confusionMatrix(data = predictions, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'LDA')

# -----------------------------------------------------------------------------
# 6. KNN
# -----------------------------------------------------------------------------

# knn with caret with cross validation to choose K
# 5-fold cross validation
trControl = trainControl(method  = "cv", number  = 5)
fit = train(RESPONSE ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:100),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = credit.train)

knn.pred = predict(fit, newdata=credit.test, type="prob")
knn.pred = ifelse(knn.pred[,2]<threshold,0,1)
knn.pred = as.factor(knn.pred)
# Confusion matrix
cm = confusionMatrix(data = knn.pred, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'KNN')

# -----------------------------------------------------------------------------
# 7. SVM
# -----------------------------------------------------------------------------

classifier = svm(formula = RESPONSE ~ .,
                data = credit.train,
                type = 'C-classification',
                kernel = 'linear',
                probability=TRUE)

y_pred = predict(classifier, newdata = credit.test, probability=TRUE)
y_pred = attr(y_pred, "probabilities")
y_pred = ifelse(y_pred[,2]<threshold,0,1)
y_pred = as.factor(y_pred)
# Confusion matrix
cm = confusionMatrix(data = y_pred, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'SVM')

# -----------------------------------------------------------------------------
# 8. NAIVE BAYES
# -----------------------------------------------------------------------------

model = naive_bayes(RESPONSE ~ ., data = credit.train, usekernel = T, prior=c(0.5,0.5))
p = predict(model, credit.test, type = 'prob')
p = ifelse(p[,2]<threshold,0,1)
p = as.factor(p)
# Confusion matrix
cm = confusionMatrix(data = p, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'NAIVE BAYES')

# -----------------------------------------------------------
# COMBINATION
# -----------------------------------------------------------

# Another option is to make not so conservative models (threshold = 0.5), and combine their predictions
# The prediction will be 1 only if ALL models predict 1
# Run all code with threshold = 0.5
new_pred = ifelse(tree.pred==0 | nn.pred==0 | glm.pred==0 | knn.pred==0 | y_pred==0 | predictions==0 | p==0, 0, 1)
pred = as.factor(new_pred)
# Confusion matrix
cm = confusionMatrix(data = pred, reference = credit.test$RESPONSE)
# Visualize 
draw_confusion_matrix(cm,'COMBINATION')
