#Analyst - Reggie Ferrell
#Project - OUSD Random Forest

#Load packages
library("ggplot2") #For data visualizations
library("dplyr") #Data manipulation (mutate,filter
library("tidyverse") #Helps transform data formats as needed
library("janitor") #Cleans variable names 
library("haven") #Load stata files into R (read_dta)
library("randomForest") #Machine learning 
library("caTools") #splits data into training and test sets
library("caret") #Helps with training, tuning, and evaluating models
library("e1071") #helos with parameter setting/tuning

setwd("/Users/rferrel/Library/CloudStorage/Box-Box/01_REL West 2022-2027/Task 4 TCTS/4.1 TCTS/4.1.16 Using Machine Learning to Predict Student Needs in OUSD/Documentation & Data/")

#Outline
#1. Import data, identify outcome variable, and turn all predictor variables into categorical data (factors = 0,1)
#2. Split source data into two datasets: 1 training and 1 testing 
#3. Perform cross validation to experiment with the best tuning parameters for the model
#4. Test your model (Optimal number of trees? Optimal number of variables?)
#5. Run your model with adjusted parameters and visualize

########## Step 1: Import data, identify outcome variable, and turn all predictor variables into categorical data (factors = 0,1)
#Import data
data <- read_dta("/Users/rferrel/Library/CloudStorage/Box-Box/01_REL West 2022-2027/Task 4 TCTS/4.1 TCTS/4.1.16 Using Machine Learning to Predict Student Needs in OUSD/Documentation & Data/ecls_subsetclean.dta") 
set.seed = 8675309	#Consistent randomization

#Global - DLimit data 
explanatory_basedemo <- data %>% select("male", "race1", "race2", "race3", "race4","race5", 
"race6", "race7", "ell_ever", "parEdlev_base1","swd_ever","pov_base","pov_near_base","parEdlev_base1",
"parEdlev_base2", "parEdlev_base3", "parEdlev_base4", "parEdlev_base5", "parEdlev_base6",
"parEdlev_base7", "parEdlev_base8", "mar_stat1","mar_stat2", "mar_stat3", "mar_stat4", "mar_stat5",
"read1","read2","read3","read4","read5","read6","read7","read8","math1","math2","math3","math4",
"math5","math6","math7","math8","math9","scien2","scien3","scien4","scien5","scien6","scien7","scien8")


summary(explanatory_basedemo$math9) #Looking at how this data is structured - Currently a numerical(continuous) value 

#Prepping the dataset 
data_prepped <- explanatory_basedemo %>% # %>% symbol is called a "pipe' in R and essentially allows code to be layered without needing to continually call the original data and/or create excess data frames. Just my coding preference. 
   mutate(across(where(~all(unique(.[!is.na(.)]) %in% c(0,1))), as.factor)) %>%  #for all variables where data is binary(0,1) turn the data type into a factor. Factor allows this data to be read as categories as opposed to integers.
  mutate(math9_pass = ifelse(math9 > 100,1,0), #Target variable: Math9_pass. Changing Math0_pass to binary factor to predict outcome. Using a random number to make the split. Just for testing purposes.
         math9_pass = as.factor(math9_pass)) %>% #Again, turning our target variable into a factor (categorical)
#filter(!is.na(math9_pass)) #Removing all NA observations from our outcome variable - Can also do this later
na.omit() %>% select(-(math9))

#Inspecting the dataset 
str(data_prepped$math9_pass) #Shows the structure of the desired variable. We want factors. 
table(data_prepped$math9_pass) #Tabulate outcome variable
sapply(data_prepped,class) #Shows the datatype of each variable 

########## Step 2: Split source data into two datasets: 1 training and 1 testing 
#Split data into testing and training datasets
sample <- sample.split(data_prepped$math9_pass, SplitRatio = .50) #Creates sample dataset that creates flag that can used to divide data into testing dataset and training dataset. Refer to caTools package for more information. SplitRatio at .50 splits our larger datast into halves. 1 for the training and another for the testing set. 
training_set <-subset(data_prepped, sample == TRUE) #Training set used to build model 
testing_set <-subset(data_prepped, sample == FALSE) #Testing set tests on the accuracy of the model


######## Step 3: Perform cross validation to experiment with the best tuning parameters for the model
#Use the train() function to evaluate model
#### Best Mtry 
trControl <- trainControl(method="cv", number=15, search="grid")
tuneGrid <- expand.grid(.mtry=c(1:15))
best_mtry <- train(math9_pass~.,
                 data = training_set,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 ntree = 100)
print(best_mtry)

# #Run the randomforest model 
randomforest_model <-  randomForest(math9_pass ~ .,data=training_set, #Orders our data to show that pass is our desired outcome and to order all variables behind it. We want all other variables to predict the first variable.
          ntree = 500,#Number of trees to grow.
          mtry = 8,#Number of variables randomly sampled as candidates at each split.
          nodesize = 1,#Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time).
          na.action = na.omit,#Removes all NAs from the dataset. Allows the model to run smoothly.
          proximity = TRUE) #Creates a table to view model outcome
 print(randomforest_model)

prediction <- predict(randomforest_model,testing_set) #Test the random forest model against the test data set 
testing_set$prediction <- prediction #Adding the predicted outcome to the testing dataset to view in one place
view(testing_set) #Allows you to look side by side at the outcome against the prediction variable
output <- table(testing_set$math9_pass, testing_set$prediction) #View how the prediction held up against the testing data
print(output) #Columns are true values, Rows are predicted values

########## Step 4: Test your model (Optimal number of trees? Optimal number of variables?)
# Will need to test on the optimal number of trees. Default is 500. 
test_trees <- data_frame(Trees = rep(1:nrow(randomforest_model$err.rate), times=3),
                        Type = rep(c("OOB","0","1"), each=nrow(randomforest_model$err.rate)),
                        Error = c(randomforest_model$err.rate[,"OOB"],
                                  randomforest_model$err.rate[,"0"],
                                  randomforest_model$err.rate[,"1"])) %>%
                mutate(Type = ifelse(Type=="0","No Pass",Type),
                       Type = ifelse(Type=="1","Passed",Type))

#Plot - Plot to see which number of trees minimizes error
ggplot(data = oob_error_trees, aes(x=Trees,y=Error))+
  geom_line(aes(color=Type))

#Select Optiminal Number of Trees
oob_error_trees <- test_trees %>% filter(Type=="OOB") ##Lowest error is seen at 275 trees, thus the optimal number of trees for this model is 275
  
########## Step 5: Run your model with adjusted parameters and visualize
randomforest_model_final <-  randomForest(math9_pass ~ .,data=training_set, #Orders our data to show that pass is our desired outcome and to order all variables behind it. We want all other variables to predict the first variable. 
                    ntree = 275,#Number of trees to grow.
                    mtry = 4 ,#Number of variables randomly sampled as candidates at each split.
                    nodesize = 1,#Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). 
                    na.action = na.omit,#Removes all NAs from the dataset. Allows the model to run smoothly. 
                    importance = TRUE,
                    proximity = TRUE) #Creates a table to view model outcome

prediction <- predict(randomforest_model_final,testing_set) #Test the random forest model against the test data set 
testing_set$prediction <- prediction #Adding the predicted outcome to the testing dataset to view in one place
view(testing_set) #Allows you to look side by side at the outcome against the prediction variable
output <- table(testing_set$math9_pass, testing_set$prediction) #View how the prediction held up against the testing data
print(output)

#Get list of variables that are most important in predicing the outcome variable
importance <- varImp(randomforest_model_final)

