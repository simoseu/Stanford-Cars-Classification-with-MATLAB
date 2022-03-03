% Load the file containing the dataset annotations
load('.\cars_annos.mat');
% ImageDatastore of the dataset
imds = imageDatastore('.\car_ims');
% Set labels of images in the imageDatastore
imds.Labels = categorical([annotations(:).class]);
% Split the dataset in 70% train and 30% validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');

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
% analyzeNetwork(net);

% Convert the trained network to a layer graph
lgraph = layerGraph(net);
% Find the names of the two layers to replace
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

% In most networks, the last layer with learnable weights is a fully connected layer.
% Replace this fully connected layer with a new fully connected layer with 
% the number of outputs equal to the number of classes in the new data set. 
% In some networks, such as SqueezeNet, the last learnable layer is a 1-by-1 
% convolutional layer instead. In this case, replace the convolutional 
% layer with a new convolutional layer with the number of filters equal to 
% the number of classes. To learn faster in the new layer than in the transferred layers, 
% increase the learning rate factors of the layer.
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
% Replace the classification layer with a new one without class labels
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer graph and zoom in on the last layers of the network
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

% Extract the layers and connections of the layer graph and select which layers to freeze
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:4) = freezeWeights(layers(1:4));
lgraph = createLgraphUsingConnections(layers,connections);

% Data augmentation helps prevent the network from overfitting and memorizing the exact details of the training images.
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,"ColorPreprocessing","gray2rgb");

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,"ColorPreprocessing","gray2rgb");

% Specify the training options
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train network using training data
net = trainNetwork(augimdsTrain,lgraph,options);

% Classify the validation images
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);


disp('Accuracy: '+string(accuracy));