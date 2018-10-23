################################################################################
##################### Group 1-10 Final ML Project ##############################
################################################################################
rm(list = ls())

##############################
### Load required packages ###
##############################
installIfAbsentAndLoad  <-  function(neededVector) {
    if(length(neededVector) > 0) {
        for(thispackage in neededVector) {
            if(! require(thispackage, character.only = T)) {
                install.packages(thispackage)}
            require(thispackage, character.only = T)
        }
    }
}
needed <- c('caret', 'splines', 'gam', "randomForest", "gbm", "glmnet", "tree", "FNN", "dplyr", "rgl")      
installIfAbsentAndLoad(needed)
# suppress warnings (rank deficient fit for test models)
oldw <- getOption("warn")
options(warn = -1)

##############################
######## Read in Data ########
##############################
clean_data <- read.csv("TrainingData.csv", stringsAsFactors = F)
clean_test_data <- read.csv(file="TestDataCSV.csv", header=TRUE, sep=",")
clean_test_data <- clean_test_data[,-1]
factorcols <- c("datetzname", "wdire", "icon", "fog", "rain", "snow", "hail", "thunder", "tornado")
clean_data[factorcols] <- lapply(clean_data[factorcols], factor)
clean_data$hum <- as.numeric(clean_data$hum)
# remove NA's
clean_data[clean_data == -9999] <- NA
clean_data <- clean_data[complete.cases(clean_data),]
clean_data$EnergyGenerated <- ifelse(clean_data$EnergyGenerated < 0, 0, clean_data$EnergyGenerated)
clean_test_data[factorcols] <- lapply(clean_test_data[factorcols], factor)
###### Karen variable preprocessing ######
set.seed(279)
uneeded <- c(11)
data = clean_data[,-uneeded]

dummies <- dummyVars(EnergyGenerated ~ ., data = data)
XVals = predict(dummies, newdata = data)
trainsetnames <- colnames(XVals)
trainsetnames <- trainsetnames[-c(23, 33, 34, 37, 38)]

nzv <- nearZeroVar(XVals)
XVals <- XVals[, -nzv]

descrCor <-  cor(XVals)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .9)
XVals <- XVals[,-highlyCorDescr]
trainsetkeptnames <- colnames(XVals)

data = cbind(XVals, "EnergyGenerated" = data$EnergyGenerated)

n = nrow(data)
set.seed(100)
trainIndices = sample(1:n, round(0.85*n))
testIndices = (1:n)[-trainIndices]
train = as.data.frame(data[trainIndices,])
test = as.data.frame(data[testIndices,])
trainX = as.data.frame(data[trainIndices,1:(ncol(data)-1)])
trainY = as.numeric(data[trainIndices, "EnergyGenerated"])
testX = as.data.frame(data[testIndices,1:(ncol(data)-1)])
testY = as.numeric(data[testIndices, "EnergyGenerated"])

# Lukie Preprocessing
d <- clean_data
t <- clean_test_data

################################################################################
################################## Modeling ####################################
################################################################################

############################################################################
#                       Shrinkage (Ridge or Lasso)                         #
############################################################################

trainX = as.matrix(data[trainIndices,1:(ncol(data)-1)])
trainY = as.matrix(data[trainIndices, "EnergyGenerated"])
testX = as.matrix(data[testIndices,1:(ncol(data)-1)])
testY = as.matrix(data[testIndices, "EnergyGenerated"])

#alpha = 0 is the ridge penalty, alpha = 1 is the lasso penalty.
#alpha can be between 0 and 1.
ridgeCVMod = cv.glmnet(trainX,trainY,alpha=0, thresh = 1e-12)
#i think this will find the best cross-validated lambda value for the lambdas
#that cv.glmnet tried.
bestLambda = ridgeCVMod$lambda[which.min(ridgeCVMod$cvm)]
ridge.mod = glmnet(trainX,trainY,alpha=0, lambda = bestLambda, thresh = 1e-12)
ridge.pred = predict(ridge.mod, s=bestLambda, newx = testX)
ridge.MSE = mean((ridge.pred-testY)^2)
#1.167706

#lasso
lassoCVMod = cv.glmnet(trainX,trainY,alpha=1, thresh = 1e-12)
bestLambda = lassoCVMod$lambda[which.min(lassoCVMod$cvm)]
lasso.mod = glmnet(trainX,trainY,alpha=0, lambda = bestLambda, thresh = 1e-12)
lasso.pred = predict(lasso.mod, s=bestLambda, newx = testX)
lasso.MSE = mean((lasso.pred-testY)^2)
#1.160582

grid = seq(0,1,0.05)
alphaMatrix = matrix(nrow= length(grid), ncol = 3, 
                     dimnames = list(list(), list("alpha", "lambda", "testMSE")))
for (i in 1:length(grid)) {
    shrinkCVMod = cv.glmnet(trainX,trainY,alpha=grid[i], thresh = 1e-12)
    bestLambda = shrinkCVMod$lambda[which.min(shrinkCVMod$cvm)]
    shrink.mod = glmnet(trainX,trainY,alpha=grid[i], lambda = bestLambda, thresh = 1e-12)
    shrink.pred = predict(shrink.mod, s=bestLambda, newx = testX)
    shrink.MSE = mean((shrink.pred-testY)^2)
    alphaMatrix[i,] = c(grid[i], bestLambda, shrink.MSE)
}

alphaMatrix[which.min(alphaMatrix[,3]),]
#      alpha      lambda     testMSE 
#0.050000000 0.005395222 1.161021207
#that lambda is a pretty small penalty (almost no penalty at all for more variables.)
#also the MSE is still pretty high (1.237) not great.  hopefully other models are better.


#####################################################################
#                              GAMS                                 #
#      includes splines, smoothing splines, and local regression    #
#####################################################################
gam1 <- gam(EnergyGenerated ~ Hour + datetzname.LA + s(tempi, 3) + 
                ns(dewpti, 5) + ns(hum, 2) + s(wspdm,2) + ns(wdird, 3) +
                ns(pressurem, 4) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy +
                s(Radiance, 4), data=train)
preds = predict(gam1, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.048473

gam2 <- gam(EnergyGenerated ~ Hour + datetzname.LA + s(tempi, 6) + 
                ns(dewpti, 6) + ns(hum, 6) + s(wspdm,4) + ns(wdird, 7) +
                ns(pressurem, 4) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy +
                s(Radiance, 8), data=train)
preds = predict(gam2, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.032495

gam3 <- gam(EnergyGenerated ~ Hour + datetzname.LA + s(tempi, 7) + 
                ns(dewpti, 8) + ns(hum, 4) + s(wspdm,8) + ns(wdird, 8) +
                ns(pressurem, 6) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy +
                s(Radiance, 10), data=train)
preds = predict(gam3, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
# 1.029385

#the local regression makes the GAM take much longer to run.
gam4 <- gam(EnergyGenerated ~ Hour + datetzname.LA + s(tempi, 7) + 
                ns(dewpti, 8) + s(hum, 4) + s(wspdm,8) + ns(wdird, 8) +
                ns(pressurem, 6) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy +
                lo(Radiance, span = 0.1), data=train)
preds = predict(gam4, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.02659

gam5 <- gam(EnergyGenerated ~ Hour + datetzname.LA + s(tempi, 20) + 
                s(dewpti,20) + s(hum, 20) + s(wspdm,20) + s(wdird, 20) +
                s(pressurem, 20) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy +
                s(Radiance, 20), data=train)
preds = predict(gam5, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.013354

gam6 <- gam(EnergyGenerated ~ s(Hour, 23) + datetzname.LA  + 
                ns(dewpti,20) + s(hum, 20) + ns(wspdm,20) + ns(wdird, 20) +
                ns(pressurem, 15) + s(tempi, 30) +
                s(Radiance, 30) + icon.clear + icon.cloudy +
                icon.mostlycloudy + icon.partlycloudy, data=train)
preds = predict(gam6, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#0.6921477


####################################################################
#                       Random Forests                             #
####################################################################
set.seed(100)
#Take out the first column of #'s below, if you would like to see the
#full tuning of the random forest.  WARNING: takes 5 hours to run.
##make a matrix to hold the OOBs for the min cost based on
##ntrees for each mtry considered.
#mtry.errors <- data.frame()
##loop through all possible mtry's.
#for (predsConsidered in seq(1, 10)) {
#  rf = randomForest(EnergyGenerated~.,data = train, ntree = 500, mtry = predsConsidered)
#  OOBErrorByNtree = rf$mse
#  minerror <- min(OOBErrorByNtree)
#  mintrees <- which.min(OOBErrorByNtree)
#  mtry.errors <- rbind(mtry.errors, c(predsConsidered, mintrees, minerror))
#  print(predsConsidered)
#} 
##note: ntree = 5000 overflowed R ("Error: cannot allocate vector of size 4.7 Gb)
##alright, started the loop above at 5:45 p.m. on 4/1/2018 finished at 11:00 (5 hours)
#
## ******
#
#bestntree <- mtry.errors[which.min(mtry.errors[,3]),2]
#bestmtry <- mtry.errors[which.min(mtry.errors[,3]),1]
##best ntree = 496, best mtry = 3
##0.2925870
#
##now rebuild that best model.
#rf <- randomForest(EnergyGenerated ~ ., data=train, ntree=bestntree, mtry=bestmtry,
#                   importance=TRUE, localImp=TRUE)

rf <- randomForest(EnergyGenerated ~ ., data=train, ntree=496, mtry=3,
                   importance=TRUE, localImp=TRUE)

sum((train$EnergyGenerated - rf$predicted)^2)/length(rf$predicted)
#0.2923477
#try it out on the test set. 
preds = predict(rf, newdata = test)
sum((test$EnergyGenerated - preds)^2)/length(preds)
#0.2941536


####################################################################
#                       kMeans Clustering                          #
####################################################################
# Temp, Humidity, and Pressure
kmeans_data <- clean_data[abs(clean_data$pressurei) < 1,]
km.out = kmeans(kmeans_data[,c(3, 7, 15)], 4, nstart = 20)
kmeans_data$cluster <- km.out$cluster
plot3d(kmeans_data[,c(3, 7, 15)], col=km.out$cluster)
avgEnergy_clust1 <- kmeans_data %>% group_by(cluster) %>% summarise(AvgEnergyGenerated = mean(EnergyGenerated))
# Hour, Temp, and Humidity
km.out = kmeans(kmeans_data[,c(1, 3, 7)], 5, nstart = 20)
kmeans_data$cluster <- km.out$cluster
plot3d(kmeans_data[,c(1,3, 7)], col=km.out$cluster)
avgEnergy_clust2 <- kmeans_data %>% group_by(cluster) %>% summarise(AvgEnergyGenerated = mean(EnergyGenerated))


####################################################################
#                       Linear Regression                          #
####################################################################
nobs <- nrow(clean_data)
trainprop <- 0.75  
train  <-  sample(nobs, trainprop * nobs)
test <- setdiff(1:nobs, train)
trainset <- clean_data[train,]
testset <- clean_data[test,]

lm.fit<- lm(EnergyGenerated ~ ., data=trainset)
lm.pred <- predict(lm.fit, testset)
MSE<- mean((testset$EnergyGenerated-lm.pred)^2)
MSE
#.704535


####################################################################
#                       Logistic Regression                        #
####################################################################
# (is power generated or not?)

generated <-rep('0', nrow(clean_data))
indices <- which(clean_data$EnergyGenerated > 0)
generated[indices] <- '1'
logisticdata <- data.frame(clean_data[1:23], generated)

nobs <- nrow(logisticdata)
trainprop <- 0.75  
train  <-  sample(nobs, trainprop * nobs)
test <- setdiff(1:nobs, train)
trainset <- logisticdata[train,]
testset <- logisticdata[test,]

glm.fit <- glm(generated~., data=logisticdata, family=binomial)
glm.pred <- predict(glm.fit, testset)
direct.pred <- which(glm.pred >0)
predicted <- rep('0', nrow(testset))
predicted[direct.pred] <- '1'
actuals <- testset$generated
glmcmat <- table(actuals, predicted)
glmcmat
sum(glmcmat[1,1],glmcmat[2,2])/sum(glmcmat)
#92% of cases correctly identified

glm.fit2 <- glm(generated~Radiance+Hour+datetzname, data=logisticdata, family=binomial)
glm.pred <- predict(glm.fit2, testset)
direct.pred <- which(glm.pred >0)
predicted <- rep('0', nrow(testset))
predicted[direct.pred] <- '1'
actuals <- testset$generated
glmcmat <- table(actuals, predicted)
glmcmat
sum(glmcmat[1,1],glmcmat[2,2])/sum(glmcmat)
#89% of cases correctly predicted with just 3 variables


####################################################################
#                       GBM Boosting                               #
####################################################################
nobs <- nrow(clean_data)
trainprop <- 0.75  
train  <-  sample(nobs, trainprop * nobs)
test <- setdiff(1:nobs, train)
trainset <- clean_data[train,]
testset <- clean_data[test,]
boostmodel = gbm(EnergyGenerated~., data = trainset,
                 distribution = "gaussian", n.trees = 100,
                 interaction.depth = 15, shrinkage = 0.01, bag.fraction = 0.5, 
                 cv.folds = 10)

prtest <- predict(boostmodel, newdata=testset, n.trees = 50)
GBM.MSE = mean((prtest-testset$EnergyGenerated)^2)
GBM.MSE
#1.11 MSE


#################################################
#       Multiple Linear Regression              #
#################################################
## first do a tree to see the most influencing factors
tree.d = tree(EnergyGenerated~.,d)
summary(tree.d)
plot(tree.d)
text(tree.d,pretty=0)
###Comment: after the visualization, we can see that Hour, Radiance, Datetzname, Temperature, and Icon are the most important factors to the EnergyGenerated

## Then fit the linear model based on the tree 
names(d)
fit.ML=lm(EnergyGenerated~Hour+tempm+Radiance, data=d)
summary(fit.ML)
yhat.ML=predict(fit.ML,data=t)
plot(yhat.ML,d$EnergyGenerated)
test.errors.ML<- mean((d$EnergyGenerated - yhat.ML)^2)
###Comment: test error is 1.35 for this model 


###########################################
#                    KNN                  #
###########################################
## first, build and tune the KNN model 
## split the training set to a train & test set to tune KNN
set.seed(5082)
d1 = d[c(1,3,4,5,6,7,8,9,10,12,13,14,15,23,24)]
trainprop <- 0.8
validateprop <- 0.2
n <- nrow(d1)
train <-sample(n, trainprop * n)
test <- setdiff(1:n,train)
trainset <- d1[train,]
testset <- d1[test,]
train.x <- trainset[-15]
train.y <- trainset[15]
test.x <- testset[-15]
test.y <- testset[15]
kset <- seq(1,19,2)
test.errors <- rep(0, length(kset))
train.errors <- rep(0, length(kset))
for(i in kset) {
    knn.pred <- knn.reg(train.x, test.x, train.y, k = i)
    test.errors[(i+1)/2] <- mean((test.y - knn.pred$pred)^2) ##this whole half thing is the error rate
    
    knn.pred <- knn.reg(train.x, train.x, train.y, k = i)
    train.errors[(i+1)/2] <- mean((train.y - knn.pred$pred)^2)    
}
plot(NULL, NULL, type='n', xlim=c(19, 1), ylim=c(0,max(c(test.errors, train.errors))), xlab='Increasing Flexibility (Decreasing k)', ylab='Mean Squared Errors', main='MSEs as a Function of Flexibiliy for KNN Regression') ##set up the graph, but now it is empty. 
lines(seq(19, 1, -2), test.errors[length(test.errors):1], type='b', col=2, pch=16) ## col=2 means color is red. 
lines(seq(19, 1, -2), train.errors[length(train.errors):1], type='b', col=1, pch=16)
legend("topright", legend = c("Test MSEs", "Train MSEs"), col=c(2, 1), cex=.5, pch=16) ## cex means magnifier. Now cex=0.75 means make it smaller. 

print(paste("Minimum Test set MSE occurred at k =", kset[which.min(test.errors)]))
print(paste("Minimum Test MSE was ", test.errors[which.min(test.errors)]))
###Comment: the test error was very good, at 0.42, with k=19


################################################################################
############################## Predict on Test Set #############################
################################################################################

# KNN 
set.seed(5078)
d.try = d[c(1,3,4,5,6,7,8,9,10,12,13,14,15,23,24)]
d.try[,1] <- as.numeric(d.try[,1])
t.try = t[c(1,3,4,5,6,7,8,9,10,12,13,14,15,23,24)]
t.try[,1] <- as.numeric(t.try[,1])
knn.pred.try <- knn.reg(d.try[1:14], t.try[1:14], d.try[15], k = 19)
KNN.MSE <- mean((t.try[15] - knn.pred.try$pred)^2)
KNN.MSE
KNN.MAE <- sum(abs(t.try[15] - knn.pred.try$pred)) / nrow(t.try[15])
KNN.MAE
KNN.AAPE <- atan(abs(t.try[15] - knn.pred.try$pred) / t.try[15])
KNN.MAAPE <- KNN.AAPE[complete.cases(KNN.AAPE),]
KNN.MAAPE <- sum(KNN.MAAPE) / nrow(t.try[15])
KNN.MAAPE                  

# Linear Regression
lm.pred <- predict(lm.fit, clean_test_data)
LiR.MSE <- mean((clean_test_data$EnergyGenerated-lm.pred)^2)
LiR.MSE
LiR.MAE <- sum(abs(clean_test_data$EnergyGenerated - lm.pred)) / length(lm.pred)
LiR.MAE
LiR.AAPE <- atan(abs(clean_test_data$EnergyGenerated - lm.pred) / clean_test_data$EnergyGenerated)
LiR.MAAPE <- LiR.AAPE[complete.cases(LiR.AAPE)]
LiR.MAAPE <- sum(LiR.MAAPE) / length(lm.pred)
LiR.MAAPE 

# Logistic Regression
generated <-rep('0', nrow(clean_test_data))
indices2 <- which(clean_test_data$EnergyGenerated > 0)
generated[indices2] <- '1'
logisticdata2 <- data.frame(clean_test_data[1:23], generated)

glm.pred <- predict(glm.fit, logisticdata2)
direct.pred <- which(glm.pred >0)
predicted <- rep('0', nrow(clean_test_data))
predicted[direct.pred] <- '1'
actuals <- logisticdata2$generated
glmcmat <- table(actuals, predicted)
glmcmat
Log.Acc <- sum(glmcmat[1,1],glmcmat[2,2])/sum(glmcmat)
Log.Acc

# Preprocessing on Test for RF and GAM
set.seed(279)
uneeded <- c(11, 20, 22)
data = clean_test_data[,-uneeded]
dummies <- dummyVars(EnergyGenerated ~ ., data = data)
XVals = predict(dummies, newdata = data)
colnames(XVals) <- trainsetnames
XVals <- XVals[,trainsetkeptnames]
data = data.frame(XVals, "EnergyGenerated" = data$EnergyGenerated)

# Random Forest
preds = predict(rf, newdata = data)
RF.MSE <- sum((data$EnergyGenerated - preds)^2)/length(preds)
RF.MSE
RF.MAE <- sum(abs(data$EnergyGenerated - preds)) / length(preds)
RF.MAE
RF.AAPE <- atan(abs(data$EnergyGenerated - preds) / data$EnergyGenerated)
RF.MAAPE <- RF.AAPE[complete.cases(RF.AAPE)]
RF.MAAPE <- sum(RF.MAAPE) / length(preds)
RF.MAAPE 

# GAMs
preds = predict(gam6, newdata = data)
GAM.MSE = sum((data$EnergyGenerated - preds)^2)/length(preds)
GAM.MSE
GAM.MAE <- sum(abs(data$EnergyGenerated - preds)) / length(preds)
GAM.MAE
GAM.AAPE <- atan(abs(data$EnergyGenerated - preds) / data$EnergyGenerated)
GAM.MAAPE <- GAM.AAPE[complete.cases(GAM.AAPE)]
GAM.MAAPE <- sum(GAM.MAAPE) / length(preds)
GAM.MAAPE 

options(warn = oldw)