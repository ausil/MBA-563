library(pacman)
p_load("recommenderlab", "arules") # recommenderlab is a library for collaborative filtering 


bb <- read.csv("BigBasket.csv") # read the data file

customer <- as(split(bb[,"Description"], bb[, "Member"]), "transactions") #Note that we used "member" here instead of "order" as we aggregate at the customer level 
# The new object "customer" is of transactions class. It saves all the items each customer bought in a list in one row
                                            
cust_mat <- as(customer,"matrix") # we convert customer objetc (of transaction class) into  matrix "cust_mat"

# Basic exploration                                            
sum(cust_mat[1,]) # Check how many unique items were bought by customer 1
dim(cust_mat) # Total rows is 106 (num of customers) and columns = 216 (total number of items)

# We will now convert this customer level transaction data into BinaryRating object.
# BinaryRating is an object type defined within recommenderlab 
# BinaryRating is for 0-1 values (such as bought vs not bought) 
# For ratings such as provided on Amazon (1-5 points), we use RealRatingMatrix object. 
# Given the nature of the data in the case study, we will use BinaryRating matrix

cust_bin <- as(cust_mat, "binaryRatingMatrix") # Converting the normal matrix into a Binary_Matrix
hist(rowCounts(cust_bin), breaks=50) # Plotting the counts of items bought by each customer

# Technical (implementation) details on the algorithm
# Data is broken into two parts: train and test - train is to fit the model, test is evaluate. 
# Available ratings for each customer in the test set will be broken into two parts:
# 1) given/ known --> to be used as input for finding similar users/ customers in the train data
# 2) withheld/ unknown --> to be used for evaluating the recommender model predictive performance

input_rating <- 1 #defines how many items to use as input using which we we find similar users for each test user
# Recommender Lab needs us to define an evaluation scheme for testing how well does the recommender 
# engine predicts what would customer would buy. 
# In this we define the method to split the data in the ration of 80/20 for training and testing. 

eval_scheme <- evaluationScheme(cust_bin, method="split", train=0.8, given=input_rating) 
# here we define "eval_scheme" as our evaluation scheme which we will call while making predictions
# to assess the predictive performance of our model. 

##############-------------- Developing Recommender Systems--------------######################
# User-Based 
UBCF_recommender <- Recommender(getData(eval_scheme, "train"), "UBCF") 
# here we define "UBCF_recommender", a User-based recommender which will be 
# trained on 80% data defined as train dataset as per our evaluation scheme "eval_scheme"

# Item-Based 
IBCF_recommender <- Recommender(getData(eval_scheme, "train"), "IBCF") 
# here we define "IBCF_recommender", an Item-based recommender which will be 
# trained on 80% data defined as train dataset as per our evaluation scheme "eval_scheme"

# Popularity-Based
POPULAR_recommender <- Recommender(getData(eval_scheme, "train"), "POPULAR")
# here we define "POPULAR_recommender", a popularity-based recommender 
# whcih recoemmends the most popular items to a customer

# Random
RANDOM_reocmmender <- Recommender(getData(eval_scheme, "train"), "RANDOM")
# This recommender generates random recommendation. This is the lowest level of performance
# if your model performs below this- we feel sorry for you  :-( 
# But don't worry that should not happen

# Generating top "n" recommendations based on the recommender systems developed above
top_recommendations_to_generate <- 5 # number of top n recommendations for a user 

# Below we predict top 5 items for customers on the basis of 
# user-based & item-based collaborative filtering, and on the basis of popularity based recommender
# systems. Finally we generate just random predictions. 

UBCF_prediction <- predict(UBCF_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate) # user-based prediction of top 5 items a customer is likely purchase in a basket based on what we see them buying 
IBCF_prediction <- predict(IBCF_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate) # item-based prediction of top 5 items a customer is likely purchase in a basket based on what we see them buying
POPULAR_prediction <- predict(POPULAR_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate)# recommending the top 5 most popular items
RANDOM_prediction <- predict(RANDOM_reocmmender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate)# recommending random items 

# Evaluating the performance of various recommenders
# Below we calculate the error matrices on the heldout (unknown) ratings of test rows  for all recommenders & we compare the error rates of both the methods. 
error_metrics <- rbind(UBCF = calcPredictionAccuracy(UBCF_prediction, getData(eval_scheme, "unknown"), given =input_rating), # performance comparison for User based vs Item based
                       IBCF = calcPredictionAccuracy(IBCF_prediction, getData(eval_scheme, "unknown"), given =input_rating),
                       Popular = calcPredictionAccuracy(POPULAR_prediction, getData(eval_scheme, "unknown"), given =input_rating),
                       Random =  calcPredictionAccuracy(RANDOM_prediction, getData(eval_scheme, "unknown"), given =input_rating))


error_metrics # Compare recommenders across range of predictive metrics 
# UBCF numbers can vary a lot -- do not get alarmed

#           TP         FP       FN       TN     precision     recall        TPR          FPR
# UBCF    4.863636 0.13636364 50.40909 159.5909 0.9727273 0.09607653 0.09607653 0.0008368356
# IBCF    3.090909 1.90909091 52.18182 157.8182 0.6181818 0.06616273 0.06616273 0.0125433433
# Popular 4.954545 0.04545455 50.31818 159.6818 0.9909091 0.09786525 0.09786525 0.0002823264
# Random  1.272727 3.72727273 54.00000 156.0000 0.2545455 0.02283946 0.02283946 0.0232687737

# The performance of  POPULAR is the best, followed by IBCF or UBCF. 
# The random recommender performs the worst as expected. 

##################-----------Making Predictions for Promotion Planning------------######################
# Predicting top "n" items for the first five customers
predict_customers <- cust_mat[1:5,]  # we will make recommendation for first five customers

predict_customers_bin <- as(predict_customers, "binaryRatingMatrix")# convert to binary matrix 

# recommendations based on UBCF
system.time(ptest <- predict(UBCF_recommender, predict_customers_bin, type="topNList", n=3)) # this recommends UBCF for the 5 customers
as(ptest, "list")  # This gives the UBCF recommendations

# recommendations based on IBCF
system.time(ptest <- predict(IBCF_recommender, predict_customers_bin, type="topNList", n=3)) # this recommends IBCF for 5 customers 
as(ptest, "list") # this gives the IBCF recommendations

# recommendations based on Popularity 
system.time(ptest <- predict(POPULAR_recommender, predict_customers_bin, type="topNList", n=3)) # this recommends IBCF for 5 customers 
as(ptest, "list") # this gives the Popular items as recommendation

# Random recommendations 
system.time(ptest <- predict(RANDOM_reocmmender, predict_customers_bin, type="topNList", n=3)) # this recommends IBCF for 5 customers 
as(ptest, "list") # this gives the random recommendations

##### Note: You may change "n" above to generate top 3, 4 or 5 recommendations. 
