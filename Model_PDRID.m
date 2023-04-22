function [PDRID] = Model_PDRID()

input_size = 512;
num_channel = 16;

layers = [

    imageInputLayer([input_size input_size 1], 'Name','Input', 'Normalization', 'none', 'Name', 'Input')
    % Important: the 'Normalization' of the image input layer should be set to 'none' (by default, Matlab removes the mean of the image).

    convolution2dLayer(3,num_channel,'NumChannels','auto','Padding','same','Name','Input Stage')
    leakyReluLayer('Name','lkrelu1')


    %% D1
    %Downsample Block
    convolution2dLayer(5,num_channel*4,'Stride',2,'Padding','same','Name','conv11')
    reluLayer('Name','relu11')
    convolution2dLayer(5,num_channel*4,'Padding','same','Name','con12')
    additionLayer(2,"Name","D1add1")

    %Encoder Block
    convolution2dLayer(5,num_channel*4,'NumChannels','auto','Padding','same','Name','conv13')
    reluLayer('Name','relu12')
    convolution2dLayer(5,num_channel*4,'NumChannels','auto','Padding','same','Name','conv14')
    additionLayer(2,"Name","D1add2")


    %% D2
    %Downsample Block
    convolution2dLayer(5,num_channel*8,'Stride',2,'Padding','same','Name','conv21')
    reluLayer('Name','relu21')
    convolution2dLayer(5,num_channel*8,'Padding','same','Name','con22')
    additionLayer(2,"Name","D2add1")

    %Encoder Block
    convolution2dLayer(5,num_channel*8,'NumChannels','auto','Padding','same','Name','conv23')
    reluLayer('Name','relu22')
    convolution2dLayer(5,num_channel*8,'NumChannels','auto','Padding','same','Name','conv24')
    additionLayer(2,"Name","D2add2")


    %% D3
    %Downsample Block
    convolution2dLayer(5,num_channel*16,'Stride',2,'Padding','same','Name','conv31')
    reluLayer('Name','relu31')
    convolution2dLayer(5,num_channel*16,'Padding','same','Name','con32')
    additionLayer(2,"Name","D3add1")

    %Encoder Block1
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv33')
    reluLayer('Name','relu32')
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv34')
    additionLayer(2,"Name","D3add2")
    
    %Encoder Block2
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv35')
    reluLayer('Name','relu33')
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv36')
    additionLayer(2,"Name","D3add3")

    %Encoder Block3
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv37')
    reluLayer('Name','relu34')
    convolution2dLayer(5,num_channel*16,'NumChannels','auto','Padding','same','Name','conv38')
    additionLayer(2,"Name","D3add4")


    %% D4
    %Downsample Block
    convolution2dLayer(5,num_channel*32,'Stride',2,'Padding','same','Name','conv41')
    reluLayer('Name','relu41')
    convolution2dLayer(5,num_channel*32,'Padding','same','Name','con42')
    additionLayer(2,"Name","D4add1")

    %Encoder Block1
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv43')
    reluLayer('Name','relu42')
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv44')
    additionLayer(2,"Name","D4add2")
    
    %Encoder Block2
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv45')
    reluLayer('Name','relu43')
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv46')
    additionLayer(2,"Name","D4add3")

    %Encoder Block3
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv47')
    reluLayer('Name','relu44')
    convolution2dLayer(5,num_channel*32,'NumChannels','auto','Padding','same','Name','conv48')
    additionLayer(2,"Name","D4add4")
    
    %Decoder Block
    convolution2dLayer(3,num_channel*32,'NumChannels','auto','Padding','same','Name','conv49')
    reluLayer('Name','relu45')
    convolution2dLayer(3,num_channel*32,'NumChannels','auto','Padding','same','Name','conv410')

    %Upsample Block
    transposedConv2dLayer(2,64,'Stride',2,'Name','up1');
    additionLayer(2,"Name","skip1")


    %% Back to D3
    %Decoder Block
    convolution2dLayer(3,num_channel*4,'NumChannels','auto','Padding','same','Name','deconv31')
    reluLayer('Name','derelu31')
    convolution2dLayer(3,num_channel*4,'NumChannels','auto','Padding','same','Name','deconv32')

    %Upsample Block
    transposedConv2dLayer(2,32,'Stride',2,'Name','up2');
    additionLayer(2,"Name","skip2")


    %% Back to D2
    %Decoder Block
    convolution2dLayer(3,num_channel*2,'NumChannels','auto','Padding','same','Name','deconv21')
    reluLayer('Name','derelu21')
    convolution2dLayer(3,num_channel*2,'NumChannels','auto','Padding','same','Name','deconv22')

    %Upsample Block
    transposedConv2dLayer(2,32,'Stride',2,'Name','up3');
    additionLayer(2,"Name","skip3")


    %% Back to D1
    %Decoder Block
    convolution2dLayer(3,num_channel*2,'NumChannels','auto','Padding','same','Name','deconv11')
    reluLayer('Name','derelu11')
    convolution2dLayer(3,num_channel*2,'NumChannels','auto','Padding','same','Name','deconv12')

    %Upsample Block
    transposedConv2dLayer(2,16,'Stride',2,'Name','up4');
    additionLayer(2,"Name","skip4")

    
    %% Output Stage
    %Decoder Block
    convolution2dLayer(3,num_channel,'NumChannels','auto','Padding','same','Name','deconv01')
    reluLayer('Name','derelu01')
    convolution2dLayer(3,num_channel,'NumChannels','auto','Padding','same','Name','deconv02')
    
    convolution2dLayer(3,1,'NumChannels','auto','Padding','same','Name','conv03')
    additionLayer(2,"Name","skip5")


    %% Output
    regressionLayer('Name','output')
];

skipConv11 = convolution2dLayer(3,num_channel*4,'Stride',2,'Padding','same','Name','SkipConv11');
skipConv21 = convolution2dLayer(3,num_channel*8,'Stride',2,'Padding','same','Name','SkipConv21');
skipConv31 = convolution2dLayer(3,num_channel*16,'Stride',2,'Padding','same','Name','SkipConv31');
skipConv41 = convolution2dLayer(3,num_channel*32,'Stride',2,'Padding','same','Name','SkipConv41');

skipConnectConv1 = convolution2dLayer(3,num_channel*4,'Padding','same','Name','SkipConnectConv1');
skipConnectConv2 = convolution2dLayer(3,num_channel*2,'Padding','same','Name','SkipConnectConv2');
skipConnectConv3 = convolution2dLayer(3,num_channel*2,'Padding','same','Name','SkipConnectConv3');
skipConnectConv4 = convolution2dLayer(3,num_channel,'Padding','same','Name','SkipConnectConv4');

PDRID = layerGraph(layers);

PDRID = addLayers(PDRID,skipConv11);
PDRID = addLayers(PDRID,skipConv21);
PDRID = addLayers(PDRID,skipConv31);
PDRID = addLayers(PDRID,skipConv41);
PDRID = addLayers(PDRID,skipConnectConv1);
PDRID = addLayers(PDRID,skipConnectConv2);
PDRID = addLayers(PDRID,skipConnectConv3);
PDRID = addLayers(PDRID,skipConnectConv4);

%D1
PDRID = connectLayers(PDRID,'lkrelu1','SkipConv11');
PDRID = connectLayers(PDRID,'SkipConv11','D1add1/in2');
PDRID = connectLayers(PDRID,'D1add1','D1add2/in2');

%D2
PDRID = connectLayers(PDRID,'D1add2','SkipConv21');
PDRID = connectLayers(PDRID,'SkipConv21','D2add1/in2');
PDRID = connectLayers(PDRID,'D2add1','D2add2/in2');

%D3
PDRID = connectLayers(PDRID,'D2add2','SkipConv31');
PDRID = connectLayers(PDRID,'SkipConv31','D3add1/in2');
PDRID = connectLayers(PDRID,'D3add1','D3add2/in2');
PDRID = connectLayers(PDRID,'D3add2','D3add3/in2');
PDRID = connectLayers(PDRID,'D3add3','D3add4/in2');

%D4
PDRID = connectLayers(PDRID,'D3add4','SkipConv41');
PDRID = connectLayers(PDRID,'SkipConv41','D4add1/in2');
PDRID = connectLayers(PDRID,'D4add1','D4add2/in2');
PDRID = connectLayers(PDRID,'D4add2','D4add3/in2');
PDRID = connectLayers(PDRID,'D4add3','D4add4/in2');

%D4 skip connection
PDRID = connectLayers(PDRID,'D3add4','SkipConnectConv1');
PDRID = connectLayers(PDRID,'SkipConnectConv1','skip1/in2');

%D3 skip connection
PDRID = connectLayers(PDRID,'D2add2','SkipConnectConv2');
PDRID = connectLayers(PDRID,'SkipConnectConv2','skip2/in2');

%D2 skip connection
PDRID = connectLayers(PDRID,'D1add2','SkipConnectConv3');
PDRID = connectLayers(PDRID,'SkipConnectConv3','skip3/in2');

%D1 skip connection
PDRID = connectLayers(PDRID,'lkrelu1','SkipConnectConv4');
PDRID = connectLayers(PDRID,'SkipConnectConv4','skip4/in2');

%Global Connect
PDRID = connectLayers(PDRID,'Input','skip5/in2');


%analyzeNetwork(PDRID);

end




