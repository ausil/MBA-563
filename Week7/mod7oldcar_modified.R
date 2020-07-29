#install.packages("pacman")
library(pacman)

# install and load all required packages
p_load(caret, dplyr, randomForest, rpart)


# Read the datafile - make sure, it is available in your working directory 
df <- read.csv("Nomis Bmod7.csv")

# check number of rows and columns
dim(df)

str(df) # check structure of df

colSums(is.na(df))
# We see that there are no values in previous Rate, thus, we will remove this column

df <- df %>% select(-c("Previous_Rate", "Approve_Date"))


########################----*********----###########################
# If you run into problem with the "select" function, 
# please run the following code (first remove the #)
# remove.packages("dplyr")
# install.packages("dplyr")                                                                                 
# library(dplyr)
########################----*********----###########################

# Since there are very few rows with NAs now, we will just drop them
df <- df[complete.cases(df),] # only keep rows where there is no misisng value



# converting variables types
df$Tier <- factor(df$Tier)
df$Partner_Bin <- factor(df$Partner_Bin)
df$Outcome <- factor(df$Outcome)
str(df)


# Filter data for used cars
df_U <- df %>% filter(Car_Type=="U")

# Remove column Car_Type
df_U <- df_U %>% select(-c("Car_Type"))


#Train Test Split 

# set random seed to replicate the results 
set.seed(111) 

# Create the train and test partitions, using the "createDataPartition" function. 
# First, we select numbers of randomly 145660(70% of 208085) numbers between 1 and 208085 and save them in an object called "rows_to_keep". 

rows_to_keep <- caret::createDataPartition(y = df_U$Outcome, p= 0.7, list = FALSE)

# We will select the rows that are in the "rows_to_keep" and these will make the "training" set. 
training <- df_U[rows_to_keep,] # Training data 

# We will select the rows from Tahoe that are NOT in the "rows_to_keep" and these will make the "testing" set. 
testing <- df_U[-rows_to_keep,] # Testing data

#Transformation and Scaling 
# first get all numeric variables 
# We shall rescale only for knn
preProcValues <- caret::preProcess(training, method = c("center", "scale"))

train_scaled <- predict(preProcValues, training)
test_scaled <- predict(preProcValues, testing)

# Proportion of "0" and '1" in readmit in train and testing datasets
prop.table(table(training$Outcome))
prop.table(table(testing$Outcome))


# KNN
# Model Fitting
system.time(
knn_model <- caret::train(form = Outcome ~., data = train_scaled[1:5000,], method = "knn", tuneLength = 15)
)
plot(knn_model)

knn_model$bestTune


# Prediction on Test Data 
test_pred_knn <- predict(knn_model, newdata = test_scaled)
# Newdata is the data on which we want to make prediction. 
# You can supply any data you want

# Predicting probabilities instead of class 
test_pred_knn_prob <- predict(knn_model, newdata = test_scaled,
                              type = "prob")

saveRDS(knn_model, "knn_model.RDS")
# Confusion Matrix
CF <- caret::confusionMatrix(test_pred_knn, test_scaled$Outcome, positive = "1")

CF$table
CF$overall[1]

knnAccuracy <- CF$overall[1]

## Random Forest
# RF
# Model Fitting
system.time(
  RF_model <- caret::train(Outcome ~., data = training[1:5000,], 
                    method = "rf", ntree = 200, tuneLength = 5)
)
plot(RF_model)

RF_model$bestTune



# Prediction on Test Data 
test_pred_RF <- predict(RF_model, newdata = testing)
# Newdata is the data on which we want to make prediction. 
# You can supply any data you want

# Predicting probabilities instead of class 
test_pred_RF_prob <- predict(RF_model, newdata = testing,
                              type = "prob")

saveRDS(RF_model,"RF_model.RDS")
# Confusion Matrix
CF <- caret::confusionMatrix(test_pred_RF, testing$Outcome, positive = "1")

CF$table
CF$overall[1]

rfAccuracy <- CF$overall[1]
## Logistics Regression
# Model Fitting
system.time(
  LogReg_model <- caret::train(Outcome~., training[1:5000,],
                    method = "glm")
)

# Prediction on Test Data 
test_pred_LogReg <- predict(LogReg_model, newdata = testing)
# Newdata is the data on which we want to make prediction. 
# You can supply any data you want

# Predicting probabilities instead of class 
test_pred_LogReg_prob <- predict(LogReg_model, newdata = testing,
                             type = "prob")

saveRDS(LogReg_model,"LogReg_model.RDS")
# Confusion Matrix
CF <- caret::confusionMatrix(test_pred_LogReg, testing$Outcome, positive = "1")

CF$table
CF$overall[1]
lrAccuracy <- CF$overall[1]
# compare all models

#Print accuracies of all models
paste("Accuracy for KNN Model is ",knnAccuracy)
paste("Accuracy for Random Forest Model is ",rfAccuracy)
paste("Accuracy for Logistic Regression Model is ",lrAccuracy)




