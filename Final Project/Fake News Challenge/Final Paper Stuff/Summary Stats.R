data <- read.csv("~/Desktop/Summary Data.csv")
head(data)
mean(data[,1])
max(data[,1])
hist(data[,1], xlab = "Article Length (in Words)", main = "Histogram of Article Lengths",
     breaks = 25)
abline(v = 700, lty = 2)

mean(data[,2], na.rm = TRUE)
hist(data[,2], xlab = "Headline Length (in Words)", main = "Histogram of Headline Lengths", 
     breaks = 15)

# build the model
data <- read.csv("~/Dropbox/CS 224n/Final Project/Fake News Challenge/Code/Fake News Comparison/results.csv")
colnames(data) <- c("Agree", "Disagree", "Discuss", "Unrelated", "isFake")
logisticMod <- glm(as.factor(isFake) ~ ., data = data, family = "binomial")


