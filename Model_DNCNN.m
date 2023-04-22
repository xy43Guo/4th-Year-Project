function [DnCNN] = Model_DNCNN()

input_size = 512;
num_channel = 32;

layers = [
    imageInputLayer([input_size input_size 1], 'Name','Input', 'Normalization', 'none')
    
    % Important: the 'Normalization' of the image input layer should be set to 'none' (by default, Matlab removes the mean of the image).
    convolution2dLayer(3,num_channel,'NumChannels','auto','Padding','same','Name','conv1')
    leakyReluLayer('Name','lkrelu1')
    
    % YOUR CODE GOES HERE

    %16 layers
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','BN2')
    leakyReluLayer('Name','lkrelu2')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','BN3')
    leakyReluLayer('Name','lkrelu3')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','BN4')
    leakyReluLayer('Name','lkrelu4')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv5')
    batchNormalizationLayer('Name','BN5')
    leakyReluLayer('Name','lkrelu5')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv6')
    batchNormalizationLayer('Name','BN6')
    leakyReluLayer('Name','lkrelu6')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv7')
    batchNormalizationLayer('Name','BN7')
    leakyReluLayer('Name','lkrelu7')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv8')
    batchNormalizationLayer('Name','BN8')
    leakyReluLayer('Name','lkrelu8')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv9')
    batchNormalizationLayer('Name','BN9')
    leakyReluLayer('Name','lkrelu9')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv10')
    batchNormalizationLayer('Name','BN10')
    leakyReluLayer('Name','lkrelu10')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv11')
    batchNormalizationLayer('Name','BN11')
    leakyReluLayer('Name','lkrelu11')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv12')
    batchNormalizationLayer('Name','BN12')
    leakyReluLayer('Name','lkrelu12')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv13')
    batchNormalizationLayer('Name','BN13')
    leakyReluLayer('Name','lkrelu13')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv14')
    batchNormalizationLayer('Name','BN14')
    leakyReluLayer('Name','lkrelu14')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv15')
    batchNormalizationLayer('Name','BN15')
    leakyReluLayer('Name','lkrelu15')
    convolution2dLayer(3,num_channel,'NumChannels',num_channel,'Padding','same','Name','conv16')
    batchNormalizationLayer('Name','BN16')
    leakyReluLayer('Name','lkrelu16')
    convolution2dLayer(3,1,'NumChannels',num_channel,'Padding','same','Name','conv17')
    
    %Residual connection
    additionLayer(2,"Name","add")
    reluLayer('Name','relu21')

    %Output
    regressionLayer('Name','output')
];

%Add input with 17th conv2d layer
DnCNN = layerGraph(layers);
DnCNN = connectLayers(DnCNN,'Input','add/in2');

%analyzeNetwork(DnCNN);

end