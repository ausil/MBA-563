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


########################-------*********--------###########################
# If you run into problem with the "select" function, 
# please run the following code (first remove the #)
# remove.packages("dplyr")
# install.packages("dplyr")                                                                                 
# library(dplyr)
########################-------*********--------###########################


# Since there are very few rows with NAs now, we will just drop them
df <- df[complete.cases(df),] # only keep rows where there is no misisng value


# converting variables types
df$Tier <- factor(df$Tier)
df$Partner_Bin <- factor(df$Partner_Bin)
df$Outcome <- factor(df$Outcome)
str(df)

# Filter data for new cars
df_N <- df %>% filter(Car_Type=="N")

# Remove column Car_Type
df_N <- df_N %>% select(-c("Car_Type"))

#Train Test Split 
#####################----------GROUP_NUMBER-------------#######################
#Remove the "#" from the line below and enter your group number
# group_number <- "enter your group number here and remove this message"

# set random seed to replicate the results 
set.seed(group_number) 

# Create the train and test partitions, using the "createDataPartition" function. 
# First, we select randomly numbers between 1 and 119059 and save them in an object called "rows_to_keep". 

rows_to_keep <- caret::createDataPartition(y = df_N$Outcome, p= 0.7, list = FALSE)

# We will select the rows that are in the "rows_to_keep" and these will make the "training" set. 
training <- df_N[rows_to_keep,] # Training data 

# We will select the rows from Tahoe that are NOT in the "rows_to_keep" and these will make the "testing" set. 
testing <- df_N[-rows_to_keep,] # Testing data

#Transformation and Scaling 
# We shall rescale only for knn
preProcValues <- caret::preProcess(training, method = c("center", "scale"))

train_scaled <- predict(preProcValues, training)
test_scaled <- predict(preProcValues, testing)


# Proportion of "0" and '1" in readmit in train and testing datasets
prop.table(table(training$Outcome))
prop.table(table(testing$Outcome))

##########################------ Group Number-------#########################
# Remove the "#" from th line below and enter your group_number 
# group_number <- "enter your group number here and remove this message"

# KNN
# Model Fitting 
system.time(
  knn_model <- caret::train(form = Outcome ~., data = train_scaled[1:5000,], 
                            method = "knn", tuneGrid = expand.grid(k = group_number))
)

# Predicting class on Test Data 
test_pred_knn <- predict(knn_model, newdata = test_scaled)
# Newdata is the data on which we want to make prediction. 
# You can supply any data you want (as long as the data has the required input variables)

# Predicting probabilities instead of class 
test_pred_knn_prob <- predict(knn_model, newdata = test_scaled,
                              type = "prob")

saveRDS(knn_model, "knn_model.RDS")
# Confusion Matrix
CF <- caret::confusionMatrix(test_pred_knn, test_scaled$Outcome, positive = "1")

CF$table
CF$overall[1]

knnAccuracy <- CF$overall[1]

