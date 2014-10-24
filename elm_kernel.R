# Kernel version
kernel_matrix1 <- function(Xtrain, kernel_par){
    XXh = rowSums(Xtrain^2) %*% matrix(1, ncol = dim(Xtrain)[1])
    omega = XXh + t(XXh) - 2*(Xtrain %*% t(Xtrain))
    omega = exp(-omega/kernel_par)
}

kernel_matrix2 <- function(Xtrain, Xtest, kernel_par){
    XXh1 = rowSums(Xtrain^2) %*% matrix(1, ncol = dim(Xtest)[1])
    XXh2 = rowSums(Xtest^2) %*% matrix(1, ncol = dim(Xtrain)[1])
    omega = XXh1 + t(XXh2) - 2*(Xtrain %*% t(Xtest))
    omega = exp(-omega/kernel_par)
}


elmTrain <- function(X, Y, kernel_par, C){
    time = proc.time()
    X = t(X)
    NumberofTrainingData = dim(X)[2]
    NumberofOutputNeurons = length(levels(as.factor(Y)))
    
    # Processing traing Labels
    temp_T = matrix(0,NumberofOutputNeurons, NumberofTrainingData)
    for(i in 1:NumberofTrainingData){
        temp_T[,i][Y[i]] = 1;
    }
    T_M = 2*temp_T - 1
    
    # training kernel and OutputWeight
    Omega_train = kernel_matrix1(t(X), kernel_par)
    OutputWeight = solve(Omega_train + diag(dim(X)[2])/C) %*% t(T_M)
    
    # prediction on the training data set
    YP = t((Omega_train) %*% (OutputWeight))
    
    # Prediction result on the training data set
    trainPred = apply(YP, 2, function(x) which(x == max(x)))
    
    # return result
    result = list(trainPred = trainPred,
                  confusion = table(Y, trainPred),
                  OutputWeight = OutputWeight,
                  time = proc.time() - time)
    
}

elmTest <- function(X, Xt, Yt, kernel_para, OutputWeight){
    time = proc.time()
    Xt = t(Xt)
    NumberofTestingData = dim(Xt)[2]
    NumberofOutputNeurons = length(levels(as.factor(Yt)))
    
    # Process target testing Labels
    temp_T = matrix(0, NumberofOutputNeurons, NumberofTestingData)
    for(i in 1:NumberofTestingData){
        temp_T[,i][Yt[i]] = 1;
    }
    Tt_M = 2*temp_T - 1
    
    # Testing kernel
    Omega_test = kernel_matrix2(X, t(Xt), kernel_par)
    YtP = t(t(Omega_test) %*% OutputWeight)
    
    # Prediction result on the testing data set
    testPred = apply(YtP, 2, function(x) which(x == max(x)))
    
    # return result
    result = list(testPred = testPred,
                  confusion = table(Yt, testPred),
                  time = proc.time() - time)
}

r = elmTrain(X, Y, kernel_par, C)
rr = elmTest(X, Xt, Yt, kernel_para, r$OutputWeight)
