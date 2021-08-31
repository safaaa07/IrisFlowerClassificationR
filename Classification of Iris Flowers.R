library(caret)
#Attach the data to the environment
data(iris)
#Rename the dataset
dataset <- iris

#80% of dataset rows for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list = FALSE)
#20% for validation
validation <- dataset[-validation_index,]
#remaining 80% for training and testing the models
dataset <- dataset[validation_index,]

#dimensions of the dataset
dim(dataset)
#list types of each attribute
sapply(dataset, class)
#first 5 rows of the data
head(dataset)
#levels for the class
levels(dataset$Species)
#summarize class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)
#summarize attribute distribution
summary(dataset)

#split input and output to visualize data
x <- dataset[,1:4]
y <- dataset[,5]
#boxplot for each attribute on one image
par(mfrow=c(1,4))
  for (i in 1:4) {
    boxplot(x[,i], main=names(iris)[i])
  }
#barplot for class breakdown
plot(y)
#Scatter plot matrix
featurePlot(x=x, y=y, plot = "ellipse")
#box and whisker plot
featurePlot(x=x, y=y, plot = "box")
#density plots for each attr by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot = "density", scales = scales)

#Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"

#lda
set.seed(7)
fit.lda <- train(Species~., data = dataset, method = "lda", metric = metric, trControl = control)
#CART
set.seed(7)
fit.cart <- train(Species~., data = dataset, method = "rpart", metric = metric, trControl = control)
#kNN
set.seed(7)
fit.knn <- train(Species~., data = dataset, method = "knn", metric = metric, trControl = control)
#SVM
set.seed(7)
fit.svm <- train(Species~., data = dataset, method = "svmRadial", metric = metric, trControl = control)
#Random Forest
set.seed(7)
fit.rf <- train(Species~., data = dataset, method = "rf", metric = metric, trControl = control)

#Accuracy of the 5 models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
#comparing accuracy
dotplot(results)

#summarize best model
print(fit.lda)

#LDA to estimate on validation dataset
predictions <- predict(fit.lda, validation)
cm <- confusionMatrix(predictions, validation$Species)
cm$table
