clear all;
 
% Third party data for evaluation can be freely downloaded from 
% https://www.nist.gov/itl/products-and-services/color-feret-database
% https://fei.edu.br/~cet/facedatabase.html
% and https://www.scface.org/
% data generation to run the method:
classNumber = 2;
samples = 1000; %samplesPerClass (training)
features = 8; 
trainingFeatureVectors.classLabels = [zeros([1,samples]), ones([1,samples])]';
mu1 = [1 1 1 1 1 1 1 1];
Sigma1 = cov (rand(20,features));
z1 = mvnrnd(mu1,Sigma1,samples);
mu2 = [1 1.5 1.8 1 1.7 1 1 0.5];
Sigma2 = cov (rand(20,features));
z2 = mvnrnd(mu2,Sigma2,samples);
trainingFeatureVectors.data = [z1;z2]; 
testSamples = 200;
testFeatureVectors.classLabels = [zeros([1,testSamples]), ones([1,testSamples])]'; %test samples for each class
z3 = mvnrnd(mu1,Sigma1,testSamples);
z4 = mvnrnd(mu2,Sigma2,testSamples);
testFeatureVectors.data = [z3;z4]; 
 
% download and add the following codes from third parties:
% download function addpath_recurse from Matlab central or add each subdirectory separately recursively:
addpath_recurse('libsvm-3.17\'); % it is needed to compile libsvm code with the Mex compiler
addpath_recurse('DimensionalityReductionCodes\'); % LPP and OLPP codes
 
% parameters:
parameters.dimensionalityReduction.options.PCARatio = 0.88; % chosen PCA ratio
parameters.dimensionalityReduction.options.ReducedDim = 5; % chosen number of dimensions of the final subspace
parameters.dimensionalityReduction.supervised = true; % supervision
options        = parameters.dimensionalityReduction.options;
supervised     = parameters.dimensionalityReduction.supervised;
 
% Define the data matrix and label matrix:
options.gnd = trainingFeatureVectors.classLabels; % training data labels
 
% Defining proper parametrization for article method
if supervised
    options.NeighborMode = 'Supervised';
end
options.WeightMode = 'Binary';
options.bLDA = 1;
 
% Dimensionality reduction:
S = constructW(trainingFeatureVectors.data, options); % FeatureVectors: a matrix with all training data
[eigvector, eigvalue, bSuccess, D] = OLPP(S, options, trainingFeatureVectors.data);
 
% project training data into the final subspace
reducedTrainingFeatures.data = trainingFeatureVectors.data * eigvector;
 
% Stores the transformation data:
transformationData.eigvector = eigvector;
transformationData.eigvalue  = eigvalue;
 
% training classifier:
model = svmtrain(trainingFeatureVectors.classLabels, full(reducedTrainingFeatures.data), '-t 0 -q');
model.trainingData = reducedTrainingFeatures.data;
 
% test samples: out-of-sample
reducedTestFeatures.data = testFeatureVectors.data * transformationData.eigvector;
 
% classification:
% removes the field with the training data:
if isfield(model, 'trainingData')
     model = rmfield(model, 'trainingData');
end
 
% classification: recognition rates
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(testFeatureVectors.classLabels, reducedTestFeatures.data, model);
disp(['Classification Recognition Rate: ', num2str(accuracy_L(1))]);

