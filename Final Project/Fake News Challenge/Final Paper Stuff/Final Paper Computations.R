require(plyr)
require(pROC)
require(randomForest)

data <- read.csv("~/Dropbox/CS 224n/Final Project/Fake News Challenge/Code/Fake News Comparison/results2.csv")
colnames(data) <- c("agree", "disagree", "discuss", "unrelated", "isReal")
colMeans(data[data$isReal == 0,])
colMeans(data[data$isReal == 1,])
logisticMod <- glm(as.factor(isReal) ~ ., data = data, family = "binomial")
summary(logisticMod)

# comparison to own headline 
data <- read.csv("~/Dropbox/CS 224n/Final Project/Fake News Challenge/Code/Fake News Comparison/comparison_to_own_headline.csv")
colnames(data) <- c("predicted.label", "source")
data <- data.frame(data)
data$predicted.label <- as.factor(data$predicted.label)
data$source <- as.factor(data$source)
table <- table(data)
table <- apply(table, 1, FUN = function(x) {x/colSums(table)})
round(table, 4)



# read in the data
data <- read.csv("~/Dropbox/CS 224n/Final Project/Fake News Challenge/Code/Fake News Comparison/results_namedEntityMatching_secondModel.csv")
colnames(data) <- c("agree", "disagree", "discuss", "unrelated", "isReal")

# print out the scaled distributions
t1 <- colSums(data[data$isReal == 0,])
t2 <- colSums(data[data$isReal == 1,])
counts <- rbind(t1, t2)[,1:4]
chisq.test(counts)
counts_scaled <- apply(counts, 1, FUN = function(x) {x/sum(x)})
round(t(counts_scaled), 4)

data_scaled <- data.frame(t(apply(data, 1, FUN = function(x) {c(x[1:4]/sum(x[1:4]), x[5])})))
colnames(data_scaled) <- c("agree", "disagree", "discuss", "unrelated", "isReal")
logisticMod <- glm(as.factor(isReal) ~ unrelated + disagree + discuss, 
                   data = data_scaled, family = "binomial",
                   weights = rowSums(data[,1:4]))
summary(logisticMod)

# cross-validate the model 
folds <- sample(1:10, size = nrow(data_scaled), replace = TRUE)
results <- rep(0, nrow(data_scaled))

sapply(1:10, FUN = function(foldNum) {
  
  print(foldNum)
  # get data to include and exclude 
  train <- which(folds != foldNum)
  holdout <- which(folds == foldNum)
  
  # build the model 
  cvMod <- glm(as.factor(isReal) ~ discuss + disagree, 
               data = data_scaled[train,], family = "binomial",
               weights = rowSums(data[,1:4])[train])
  #  cvMod <- randomForest(y = as.factor(data_scaled[train,]$isReal), x = data_scaled[train,1:4])
  
  # make predictions
  #  results[holdout] <<- predict(cvMod, newdata = data_scaled[holdout,1:4], type = "prob")[,2]
  results[holdout] <<- predict(cvMod, newdata = data_scaled[holdout,], type = "response")
})
plot(roc(response = data_scaled$isReal, predictor = results))
roc(response = data_scaled$isReal, predictor = results)
