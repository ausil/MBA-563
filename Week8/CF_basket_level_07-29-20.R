library(pacman)
p_load("recommenderlab", "arules") # recommenderlab is a library for collaborative filtering 


bb <- read.csv("Bigbasket.csv") # read the data file

trans <- as(split(bb[,"Description"], bb[,"Order"]), "transactions") # Convert the data into "transaction" object

# The new object "trans" is of transactions class. It saves all the items in each transaction in a list in one row

mat <- as(trans,"matrix") # We will now convert this transaction data into a matrix to convert into BinaryRating object a native for
# Recommender lab. Binary Rating is for 0-1 values and for ratings such as 0-5, we use RealRatingMatrix object. 

sum(mat[1,]) # Check how many unique items were bought in transaction 1
dim(mat) # Total rows is 8387 (number of transactions) and columns = 216 (total number of items)

max(rowSums(mat)) # 28 maximum number of items in any transaction (row) is 28
min(rowSums(mat)) # 1 maximum number of items in any transaction (row) is 28 (this puts a restriction on "given")

# We will now convert this customer level transaction data into a matrix to convert into BinaryRating object.
# BinaryRating is an object type defined within recommenderlab 
# BinaryRating is for 0-1 values (such as bought vs not bought) 
# For ratings such as provided on Amazon (1-5 points), we use RealRatingMatrix object. 
# Given the nature of transaction data, we will use BinaryRating matrix

bin_mat <- as(mat, "binaryRatingMatrix") # Converting the normal matrix into a Binary_Matrix
hist(rowCounts(bin_mat), breaks=50) # Plotting the counts of items in the baskets (transactions)


# Available ratings in each row in the test set will be broken into two parts:
# 1) given/ known --> to be used as input for finding similar users in the train data
# 2) withheld/ unknown --> to be used for evaluating the recommender model predictive performance

input_rating <- 1 #defines how many items to assume are given (here set equal to 1)
  # Recommender Lab needs us to define an evaluation scheme for testing how well does the recommender 
  # engine predicts what would customer would buy. 
  # In this we define the method to split the data in the ration of 80/20 for training and testing. 
  
eval_scheme <- evaluationScheme(bin_mat, method="split", train=0.8, given=input_rating) 
# here we define "eval_scheme" as our evaluation scheme which we will call while making predictions. 
# to assess the predictive performance of our model. 
##############-------------- Developing Recommender Systems--------------######################
UBCF_recommender <- Recommender(getData(eval_scheme, "train"), "UBCF") 
# here we define "UBCF_recommender", a User-based recommender which will be 
# trained on 80% data defined as train dataset as per our evaluation scheme "eval_scheme"

IBCF_recommender <- Recommender(getData(eval_scheme, "train"), "IBCF") 
# here we define "IBCF_recommender", an Item-based recommender which will be 
# trained on 80% data defined as train data set as per our evaluation scheme "eval_scheme"

POPULAR_recommender <- Recommender(getData(eval_scheme, "train"), "POPULAR")
# here we define "POPULAR_recommender", a popularity based recommender 
# which recommends the most popular item toa customer

RANDOM_reocmmender <- Recommender(getData(eval_scheme, "train"), "RANDOM")
# This recommender generates random recommendation. This is the lowest level of performance


top_recommendations_to_generate <- 5 # number of top n recommendations 
# Below we predict top 5 items on the basis of 
# user-based and item-based collaborative filtering and on the basis of popularity and finally just random predictions


# Recommendations based on UBCF
UBCF_prediction <- predict(UBCF_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate) # user-based prediction of top 5 items a customer is likely purchase in a basket based on what we see them buying 

# Recommendations based on IBCF
IBCF_prediction <- predict(IBCF_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate) # item-based prediction of top 5 items a customer is likely purchase in a basket based on what we see them buying

# Recommendations based on Popularity 
POPULAR_prediction <- predict(POPULAR_recommender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate)

# Random recommendations 
RANDOM_prediction <- predict(RANDOM_reocmmender, getData(eval_scheme, "known"), type="topNList", n=top_recommendations_to_generate)


# Below we calculate the error matrices on the heldout (unknown) ratings of test rows  for all recommenders & we compare the error rates of both the methods. 
error_metrics <- rbind(UBCF = calcPredictionAccuracy(UBCF_prediction, getData(eval_scheme, "unknown"), given =input_rating), # performance comparison for User based vs Item based
                       IBCF = calcPredictionAccuracy(IBCF_prediction, getData(eval_scheme, "unknown"), given =input_rating),
                       Popular = calcPredictionAccuracy(POPULAR_prediction, getData(eval_scheme, "unknown"), given =input_rating),
                       Random =  calcPredictionAccuracy(RANDOM_prediction, getData(eval_scheme, "unknown"), given =input_rating))


error_metrics # # Compare recommenders across range of predictive metrics 


#             TP       FP       FN       TN     precision    recall       TPR        FPR
# UBCF    0.8623361 4.125745 4.286651 205.7253 0.17287933 0.1879523 0.1879523 0.01962663
# IBCF    1.3057211 3.681764 3.843266 206.1692 0.26148568 0.2854027 0.2854027 0.01749627
# Popular 1.5715137 3.428486 3.577473 206.4225 0.31430274 0.3351998 0.3351998 0.01628147
# Random  0.1269368 4.873063 5.022050 204.9779 0.02538737 0.0221499 0.0221499 0.02322077

# The performance of  POPULAR and IBCF are better. UBCF follows.
# The random recommender performs the worst as expected. 

##################-----------Making Predictions for Promotion Planning------------######################
predict_mat <- mat[1:5,]  # we will make recommendation for first five rows

rtest <- as(predict_mat, "binaryRatingMatrix") 

system.time(ptest <- predict(UBCF_recommender, rtest, type="topNList", n=3)) # this recommends UBCF for the 5 customers
# Recommendations based on UBCF
as(ptest, "list")  # This gives the UBCF recommendations

system.time(ptest <- predict(IBCF_recommender, rtest, type="topNList", n=3)) # this recommends IBCF for 5 customers 
# Recommendations based on IBCF
as(ptest, "list") # this gives the IBCF recommendations

# Recommendations based on Popularity 
system.time(ptest <- predict(POPULAR_recommender, rtest, type="topNList", n=3)) # this recommends IBCF for 5 customers 
as(ptest, "list") # this gives the Popular items as recommendation

# Random recommendations 
system.time(ptest <- predict(RANDOM_reocmmender, rtest, type="topNList", n=3)) # this recommends IBCF for 5 customers 
as(ptest, "list") # this gives the random recommendations

##### Note: You may change "n" above to generate top 3, 4 or 5 recommendations. 

