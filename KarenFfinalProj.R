##################### Group 1-10 Final ML Project ##############################
################################################################################
rm(list = ls())
###################################################################
### Functions
###################################################################
installIfAbsentAndLoad  <-  function(neededVector) {
  if(length(neededVector) > 0) {
    for(thispackage in neededVector) {
      if(! require(thispackage, character.only = T)) {
        install.packages(thispackage)}
      require(thispackage, character.only = T)
    }
  }
}
##############################
### Load required packages ###
##############################
# 
needed <- c('caret', 'splines', 'gam', 'glmnet','randomForest')      
installIfAbsentAndLoad(needed)

setwd("C:\\Users\\karen\\Documents\\MachineLearning\\MachineLearningII\\final")
clean_data <- read.csv("TrainingData.csv", stringsAsFactors = F)

set.seed(279)
uneeded <- c(11)
data = clean_data[,-uneeded]

#trying to use the caret stuff from that seminar.
dummies <- dummyVars(EnergyGenerated ~ ., data = data)
XVals = predict(dummies, newdata = data)

nzv <- nearZeroVar(XVals)
XVals <- XVals[, -nzv]

descrCor <-  cor(XVals)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .9)
XVals <- XVals[,-highlyCorDescr]

#raised an error, because no lin combos left at this point in preprocessing.
#comboInfo <- findLinearCombos(XVals)
#XVals <- XVals[, -comboInfo$remove]

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
#haven't tried running this yet, waiting for rf to finish looping...
gam1 <- gam(EnergyGenerated ~ Hour + datetznameLA + s(tempi, 3) + 
              ns(dewpti, 5) + ns(hum, 2) + s(wspdm,2) + ns(wdird, 3) +
              ns(pressurem, 4) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy +
              s(Radiance, 4), data=train)
preds = predict(gam1, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.048473

gam2 <- gam(EnergyGenerated ~ Hour + datetznameLA + s(tempi, 6) + 
              ns(dewpti, 6) + ns(hum, 6) + s(wspdm,4) + ns(wdird, 7) +
              ns(pressurem, 4) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy +
              s(Radiance, 8), data=train)
preds = predict(gam2, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.032495

gam3 <- gam(EnergyGenerated ~ Hour + datetznameLA + s(tempi, 7) + 
              ns(dewpti, 8) + ns(hum, 4) + s(wspdm,8) + ns(wdird, 8) +
              ns(pressurem, 6) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy +
              s(Radiance, 10), data=train)
preds = predict(gam3, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
# 1.029385

#the local regression makes the GAM take much longer to run.
gam4 <- gam(EnergyGenerated ~ Hour + datetznameLA + s(tempi, 7) + 
              ns(dewpti, 8) + s(hum, 4) + s(wspdm,8) + ns(wdird, 8) +
              ns(pressurem, 6) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy +
              lo(Radiance, span = 0.1), data=train)
preds = predict(gam4, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.02659

gam5 <- gam(EnergyGenerated ~ Hour + datetznameLA + s(tempi, 20) + 
              s(dewpti,20) + s(hum, 20) + s(wspdm,20) + s(wdird, 20) +
              s(pressurem, 20) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy +
              s(Radiance, 20), data=train)
preds = predict(gam5, newdata = test)
MSE = sum((test$EnergyGenerated - preds)^2)/length(preds)
MSE
#1.013354

gam6 <- gam(EnergyGenerated ~ s(Hour, 23) + datetznameLA  + 
              ns(dewpti,20) + s(hum, 20) + ns(wspdm,20) + ns(wdird, 20) +
              ns(pressurem, 15) + s(tempi, 30) +
              s(Radiance, 30) + iconclear + iconcloudy +
              iconmostlycloudy + iconpartlycloudy + datetznameLA, data=train)
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

#since we have a separate test test & rf uses OOB, maybe should have
#given it all of the data to train on?  would also have been nice to 
#try a few more ntrees, but would take too long to run.
