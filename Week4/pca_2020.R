######################################PCA############################################
getwd() # Check the working directory. You should keep your data file in this folder
# setwd("mention the path of your folder") # this allows you to set a folder of your choice 
# as the working directory

# Universities data
university <- read.csv("Universities.csv") # read data from the file in the working directory
pca_univ <- prcomp(university[,c(2:6)], center = TRUE, scale. = TRUE) # selecting columns 3-7 for PCA
PCs <- pca_univ$x # all components are saved in the files PCs
university_pca <- cbind(university,PCs) # merge all data 
summary(pca_univ)# output of PCA model 

# Regression Model to predict Graduation Rate on all 5 variables (without PCA)
model <- lm(GradRate~ SAT+TOP10+Accept+SFRatio+Expenses, data = university_pca)
summary(model)
# Multiple R-squared:  0.7104
# Adjusted R-squared:  0.6342


# Regression Model to predict Graduation Rate based on 1st component only
model_pca1 <-lm(GradRate~ PC1, data = university_pca)
summary(model_pca1)
# Multiple R-squared:  0.5414 
# Adjusted R-squared:  0.5215

# Regression Model to predict Graduation Rate based on 1st and 2nd components
model_pca2 <-lm(GradRate~ PC1+PC2, data = university_pca)
summary(model_pca2)
# Multiple R-squared:  0.6705 (0.6705/0.7104 = 0.9438345) 
# Adjusted R-squared:  0.6405



###################################################################

