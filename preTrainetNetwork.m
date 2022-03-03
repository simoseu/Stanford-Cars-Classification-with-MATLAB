% Load the file containing the dataset annotations
load('.\cars_annos.mat');
% ImageDatastore of the dataset
imds = imageDatastore('.\car_ims');
% Set labels of images in the imageDatastore
imds.Labels = categorical([annotations(:).class]);
% Split the dataset in 70% train and 30% test
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomize');

% Display some sample images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
% Start time
tic
% Load the net (uncomment the net you want to use)
net = resnet18;
% net = resnet50;
% net = resnet101;
% net = vgg16;
% net = alexnet;
% net = inceptionresnetv2;

% Input size of the net
inputSize = net.Layers(1).InputSize;

% See the structure of the net 
analyzeNetwork(net);

% Augmented datastore for the imdses with the color preprocessing for
% non-rgb images
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'ColorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize,imdsTest,'ColorPreprocessing','gray2rgb');

% Select layer (uncomment based on the net you are using)
layer = 'pool5'; % resnet-18 and resnet-101
% layer = 'global_average_pooling2d_2'; % resnet-50
% layer = 'fc7' % vgg16 and alexnet

% Feature extraction 
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% Labels of train and test images
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% Required conversions
YTrain = double(YTrain(:,1)) -1;
YTest = double(YTest(:,1)) -1;
featuresTrain = sparse(double(featuresTrain));
featuresTest = sparse(double(featuresTest));

% Classification using liblinear functions
model = train(YTrain, featuresTrain, '-s 2');

% At the end will be printed the accuracy of classify the test set
YPred = predict(YTest, featuresTest, model);

% At the end will be printed the accuracy of the classify the training set
YPredTrain = predict(YTrain, featuresTrain, model);

% Display some images with predicted labels
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(label)
end

time = toc;
diff = numel(find(YPred~=YTest));
[M,N] = size(YPred);
tp = M-diff;
accuracy = round(mean(YPred == YTest)*100,2);
disp('Accuracy: '+string(accuracy)+"% - Time Elapsed: "+time+" s - True Positive vs Total: "+tp+"/"+M);