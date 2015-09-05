function [expansionCoefficient,centerSet, biasTerm,learningCurve] = ...
    RAN(trainInput,trainTarget,testInput,testTarget,delta_max, delta_min,...
    tau, overlapFactor, stepSize,tolerancePredictError,flagLearningCurve)
%Function sparseKLMS1:   resource allocating networks with novel criteria
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Input:
%trainInput:        input signal inputDimension*trainSize, inputDimension is the input dimension and trainSize is the number of
%                   training data
%trainTarget:       desired signal for training trainSize*1
%testInput:         testing input, inputDimension*testSize, testSize is the number of the test data
%testTarget:        desired signal for testing testSize*1
%typeKernel:        'Gauss', 'Poly'
%paramKernel:       h (kernel size) for Gauss and p (order) for poly
%stepSizeFeatureVector:     learning rate for kernel part
%stepSizeWeightVector:      learning rate for linear part, set to zero to disable
%stepSizeBias:              learning rate for bias term, set to zero to disable
%flagLearningCurve:         control if calculating the learning curve
%toleranceDistance:         tolerance for the closeness of the new data to the dictionary
%tolerancePredictError:     tolerance for the apriori error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Output:
%expansionCoefficient:      onsisting of coefficients of the kernel
%                           expansion with the stepSizeFeatureVector
%weightVector:              the linear coefficients
%biasTerm:                  the bias term
%learningCurve:             trainSize*1 used for learning curve
%dictionaryIndex:           index of bases used in the kernel expansion in
%                               the training set

%delta_max, delta_min, tau, overlapFactor

% memeory initialization
trainSize = length(trainTarget);
testSize = length(testTarget);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
    learningCurve(1) = mean(testTarget.^2); 
else
    learningCurve =[];
end

% n=1 init
delta = delta_max;
predictionError = trainTarget(1);
expansionCoefficient = predictionError;

% dictionary
centerSet = trainInput(:,1);
dictSize = 1;
kernelSize = overlapFactor*delta;

biasTerm = 0;


% start
for n=2:trainSize
    % training
       
    % comparing the distance between trainInput(:,n) and the centerSet
	dis = sum((trainInput(:,n)*ones(1,dictSize) - centerSet).^2,1);
		
    distance2dictionary = sqrt(min(dis));
	
	firstLayerOutput = exp(- dis./(kernelSize.^2));
	
	networkOutput = expansionCoefficient*firstLayerOutput' + biasTerm;

	predictionError = trainTarget(n) - networkOutput;
	
    if (distance2dictionary > delta && abs(predictionError) > tolerancePredictError)
		% updating
		dictSize = dictSize + 1;
		centerSet(:,dictSize) = trainInput(:,n);
		expansionCoefficient(dictSize) = predictionError;
		kernelSize(dictSize) = overlapFactor*distance2dictionary;
	
	else
		for jj = 1:dictSize
			centerSet(:,jj) = centerSet(:,jj) + 2*stepSize*(trainInput(:,n) - centerSet(:,jj))*firstLayerOutput(jj)*predictionError*expansionCoefficient(jj)/kernelSize(jj);
		end
		expansionCoefficient = expansionCoefficient + stepSize*predictionError*firstLayerOutput;
		biasTerm = biasTerm + stepSize*predictionError;
		
	end

    if flagLearningCurve == 1
        % testing
        y_te = zeros(testSize,1);
        for jj = 1:testSize
            %ii = 1:dictSize;
            y_te(jj) = expansionCoefficient*(exp(-sum((testInput(:,jj)*ones(1,dictSize)-centerSet).^2,1)./(kernelSize.^2)))'...
				 + biasTerm;
        end
        err = testTarget - y_te;
        learningCurve(n) = mean(err.^2);
	end
	if (delta > delta_min)
		delta = delta*exp(-1/tau);
	end
	
end

return

