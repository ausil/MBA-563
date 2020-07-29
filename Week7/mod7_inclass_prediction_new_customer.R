
##############################################################
#install.packages("pacman")
library(pacman)

# install and load all required packages
p_load(caret, dplyr, randomForest, rpart, rattle)


# Read the RF model provided by us
RF_model <- readRDS("RF_model.RDS") 


# We are providing you new customer applications that are accepted by e-car
new_customers <-  readRDS("new_customers.RDS")


##########################------ Group Number-------#########################
# Remove the "#" from th line below and enter your group_number 

group_number <- 5 # defined temporarily; you will fill yours. 

# Here you are subsetting the file to find the customer whose ID matches 
# with your group. 
my_customer_RF <- new_customers %>% filter(Cust_ID == group_number)

# converting the categorical variables to factors
my_customer_RF$Tier <- factor(my_customer_RF$Tier)
my_customer_RF$Partner_Bin <- factor(my_customer_RF$Partner_Bin)
my_customer_RF$Outcome <- factor(my_customer_RF$Outcome)


# Predicting class of my_customer based on the provided RF model
pred_RF <- predict(RF_model, newdata = my_customer_RF[, -1]) # removing the customer ID column 
# Newdata is the data on which we want to make prediction. 
# You can supply any data you want (as long as the data has the required input variables)

# Predicting probabilities instead of class. This gives two probabilities: for class 0 and class 1.
# Both the probabilities add up to 100%. 
pred_RF_prob <- predict(RF_model, newdata = my_customer_RF[, -1], # we removed the ID column 
                         type = "prob")

# Adding rates to dataframe of predicted probabilities 
pred_RF_prob$Rate <- my_customer_RF$Rate

# We are interested in finding the probability of Acceptance (Outcome =1), which is in column 2. 
max_prob_row <- which.max(pred_RF_prob[,2])# Here, we are asking in which row is column2 value max.
# The row with the highest probability of acceptance is saved as max_prob_row

max_prob_rate <- pred_RF_prob[max_prob_row, 3]# Now we are asking R to give the value of Rate at which 
# acceptance is maximized. 

max_prob_rate # printing the rate with the highest probability 

# Now think whether you will you will choose the max_prob?
