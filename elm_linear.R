#linear version
elmTrain <- function(X, Y, L, C){
    time = proc.time()
    
    # parameters setting
    X = t(X)
    NumberofTrainingData = dim(X)[2]
    NumberofInputNeurons = dim(X)[1]
    NumberofOutputNeurons = length(levels(as.factor(Y)))
    NumberofHiddenNeurons = L
    
    # Processing traing Labels
    temp_T = matrix(0,NumberofOutputNeurons, NumberofTrainingData)
    for(i in 1:NumberofTrainingData){
        temp_T[,i][Y[i]] = 1;
    }
    T_M = 2*temp_T - 1
    
    # Randomly generate input weights InputWeight (w_i) 
    # and BiasofHiddenNeurons (b_i) 
    temp = runif(NumberofHiddenNeurons * NumberofInputNeurons) * 2 -1
    InputWeight = matrix(temp,NumberofHiddenNeurons, NumberofInputNeurons)
    BiasofHiddenNeutrons = runif(NumberofHiddenNeurons)   
    tempH = InputWeight %*% X + matrix(rep(BiasofHiddenNeutrons, NumberofTrainingData), 
                                       nrow = NumberofHiddenNeurons, byrow = F)
    
    # Active function here use default sigmoid
    # apply the active function to training data
    H = 1 / (1 + exp(- tempH ))
    rm(tempH) # remove tempH
    
    # OutputWeight = ginv(t(H), tol = sqrt(.Machine$double.eps)) %*% t(T_M)
    OutputWeight = solve(diag(dim(H)[1])/C + H %*% t(H)) %*% H %*% t(T_M)
    
    # prediction on the training data set
    YP = t(t(H) %*% OutputWeight)
    # Prediction result on the training data set
    trainPred = apply(YP, 2, function(x) which(x == max(x)))
    
    # return result
    result = list(trainPred = trainPred,
                  confusion = table(Y, trainPred),
                  OutputWeight = OutputWeight,
                  InputWeight = InputWeight,
                  BiasofHiddenNeutrons = BiasofHiddenNeutrons,
                  time = proc.time() - time)
}

elmTest <- function(Xt, Yt, L, InputWeight, OutputWeight, BiasofHiddenNeutrons){
    time = proc.time()
    Xt = t(Xt)
    
    # parameter setting
    NumberofTestingData = dim(Xt)[2]
    NumberofInputNeurons = dim(Xt)[1]
    NumberofOutputNeurons = length(levels(as.factor(Yt)))
    NumberofHiddenNeurons = L
    
    # Process target testing Labels
    temp_T = matrix(0, NumberofOutputNeurons, NumberofTestingData)
    for(i in 1:NumberofTestingData){
        temp_T[,i][Yt[i]] = 1;
    }
    Tt_M = 2*temp_T - 1
    
    # apply the same active function to testing data
    tempH = InputWeight %*% Xt + matrix(rep(BiasofHiddenNeutrons, NumberofTestingData), 
                                        nrow = NumberofHiddenNeurons, byrow = F)
    Ht = 1 / (1 + exp(- tempH ))
    
    YtP = t(t(Ht) %*% OutputWeight)
    # Prediction result on the testing data set
    testPred = apply(YtP, 2, function(x) which(x == max(x)))
    
    # return result
    result = list(testPred = testPred,
                  confusion = table(Yt, testPred),
                  time = proc.time() - time)
    
}

r = elmTrain(X, Y, L = 2000, C = 300)
rr = elmTest(Xt, Yt, L = 2000, InputWeight = r$InputWeight, 
              OutputWeight = r$OutputWeight, BiasofHiddenNeutrons = r$BiasofHiddenNeutrons)