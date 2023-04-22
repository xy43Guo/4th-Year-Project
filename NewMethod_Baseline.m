close all;
clear;
clc;

addpath('../utils/');
addpath('../utils/lib/');
addpath('..');
run('utils/lib/irt/setup.m');

%Determine the steps the whole program needs to run
doTrain =   false;  %If train the DnCNN network
doAdd   =   false;  %If generate noisy imageDatastore 
docheck =   false;  %If Check the performance of DnCNN network
dounzip =   false;  %If unzip the dataset

file_model = 'dncnn.mat';

%Set the training parameters
WorkerLoadNum = 4;
minibatchSize = 8;
maxEpochs = 10;
initLearningRate = 0.001;
DropPeriod = 2;
DropFactor = 0.1;
Environment = 'auto';


if dounzip
    unzip('dataset_single.zip','../');
    fprintf('Successfully Unzip!');
end

%% 1. Create ImageDatastores for training DnCNN

%Creat the folder for generated noisy images
if not(isfolder('Noisyimage'))
    mkdir('Noisyimage');
end

read_dirpath = 'dataset_single/trainingset/';  % Adjust this accordingly
read_files = dir(fullfile(read_dirpath, '*.fits'));   % pattern to match filenames.
write_dirpath = 'Noisyimage/';

if doAdd
    for i = 1: numel(read_files)
        disp(i)
        filename = read_files(i).name; % Read an image file from the directory
        read_filepath = fullfile(read_dirpath, read_files(i).name);
        img = fitsread(read_filepath);

        nosiy_img = cell2mat(addNoise(img));

        filename(end-3:end) = 'fits';               
        write_filename = [write_dirpath filename];  % fullfile name to write
        
        fitswrite(nosiy_img,write_filename);
    end
end 


%% Noisy images Test
% write_files = dir(fullfile(write_dirpath, '*.fits'));   % pattern to match filenames.
% read_filepath = fullfile(read_dirpath, read_files(99).name);
% im_o = fitsread(read_filepath);
% 
% write_filepath = fullfile(write_dirpath, write_files(99).name);
% im_n = fitsread(write_filepath);
% 
% figure(1), subplot(1,2,1), imagesc(im_o), colormap gray;
% title('Original image')
% 
% figure(1), subplot(1,2,2), imagesc(im_n), colormap gray;
% title('Noisy image')

%%
patchSize = [512 512];
patchNum = 1;

%Original images dataset
digitDatasetPath = fullfile(read_dirpath, '*.fits');
imds = imageDatastore(digitDatasetPath,'LabelSource','foldernames',  "ReadFcn", @fitsread,'ReadSize',100);
[imds_train,imds_val] = splitEachLabel(imds,0.95);

%Nosiy version images dataset
digitDatasetPath_NI = fullfile(write_dirpath, '*.fits');
imds_NV = imageDatastore(digitDatasetPath_NI,'LabelSource','foldernames',  "ReadFcn", @fitsread,'ReadSize',100);
[NV_train,NV_val] = splitEachLabel(imds_NV,0.95);

% patches_train = combine(NV_train,imds_train);
% patches_val = combine(NV_val,imds_val);

patches_train = randomPatchExtractionDatastore(NV_train, imds_train,patchSize, ...
                                         'PatchesPerImage',patchNum);
patches_val = randomPatchExtractionDatastore(NV_val, imds_val,patchSize, ...
                                         'PatchesPerImage',patchNum);

% exampleData = preview(patches_train);
% inputs = exampleData(:,1);
% responses = exampleData(:,2);
% minibatch = cat(2,inputs, responses);
% montage(minibatch, Size = [2 2]);

% a2 = read(patches_train);
% inputs = a2.InputImage;
% responses = a2.ResponseImage;
% figure, imagesc(inputs{66,1}), colormap gray;
% title('Denoised image')
% figure, imagesc(responses{66,1}), colormap gray;
% title('Test image');

%% 2. PDRID architecture and implementation

DNCNN = Model_DNCNN();

%analyzeNetwork(PDRID);

% Do not forget to check and visualize your network architecture using
% built-in function 'analyzeNetwork'

%% 3. Training the DnCNN
% Here you will train your DnCNN network on pairs of noisy and groundtruth
% patches with the above created ImageDatastores. It is highly
% recommended that you implement the training in a separate file and to
% save your training plots and trained network.
if not(isfolder('model'))
    mkdir('model')
end


if doTrain   
           
    rootDir = 'model';

    options = trainingOptions('adam', ...
                              'InitialLearnRate',initLearningRate, ...
                              'LearnRateSchedule','piecewise', ...
                              'LearnRateDropPeriod', DropPeriod, ...
                              'LearnRateDropFactor', DropFactor, ...
                              'L2Regularization',0.0001 , ...
                              'MiniBatchSize',minibatchSize, ...
                              'MaxEpochs',maxEpochs, ...
                              'Plots','training-progress', ...
                              'WorkerLoad',WorkerLoadNum, ...
                              'ExecutionEnvironment', Environment, ...
                              'Shuffle','every-epoch', ...
                              'ValidationData', patches_val, ...
                              'ValidationFrequency',50, ...
                              'Verbose',true,...
                              'VerboseFrequency',50,...
                              'CheckPointPath',rootDir);
    
    dncnn = trainNetwork(patches_train, DNCNN, options);
    save('dncnn','dncnn');

else

    load(file_model);

end

%%
if docheck
    disp("Doing the check!")
    testimgs = read(imds_NV);                                                                 
    oris = read(imds);
    
    montage([testimgs(95:100) oris(95:100)]);
    
    
    im_test = testimgs(96);
    im_ori = oris(96);
    
    pred = predict(dncnn, im_test{1,1});
    
    figure, imagesc(pred), colormap gray;
    title('Denoised image')
    figure, imagesc(im_ori{1,1}), colormap gray;
    title('Original image')
    figure, imagesc(im_test{1,1}), colormap gray;
    title('Test image')
end

%% 5. M3 implementation ...
% You will here implement the PnP-ADMM algorithm. It is highly
% recommended that you implement your algorithm in a separate file as a
% function with possibilities to pass the required inputs and parameters.
options.rel_tol = 1e-4;
options.rel_tol2 = 1e-4;


FilePath = 'dataset_single/testingset2/';%qiang
read_files = dir(fullfile(FilePath, '*.fits'));
img_num = length(read_files);

for i = 1:14

    options.max_iter = 15;
    options.delta = 0.08;

    %disp('=====================================================================================================================================================================================')
    %disp('=====================================================================================================================================================================================')
    fprintf('Caculating the %dth image!\n',i);

    image_name=read_files(i).name;
    image = fitsread(strcat(FilePath,image_name));
    
    reCoef = 1;
    temp = 0;
    
    myRange = getrangefromclass(image(1));
    newMax = myRange(2);
    newMin = myRange(1);
    image = (image - min(image(:)))*(newMax - newMin)/(max(image(:)) - min(image(:))) + newMin;

    tic
    while temp ~= 1

        if temp == 2
            options.delta = options.delta*0.08;
            options.max_iter = 10;
        else
            reCoef = reCoef*0.9;
        end

        [xsol, bp_y, t, temp] = PnP(dncnn, image, options, reCoef, i);
    end
    execuateTime = toc;
   
%     maxValue = double(max(max(xsol)));
%     xsol = double(xsol)/maxValue;

%     myRange = getrangefromclass(xsol(1));
%     newMax = myRange(2);
%     newMin = myRange(1);
%     xsol = (xsol - min(xsol(:)))*(newMax - newMin)/(max(xsol(:)) - min(xsol(:))) + newMin;
    
    rsnr = 20*log10(norm(image(:))/norm(image(:)-xsol(:)));
    img1 = double(image);
    ssimval= ssim(img1,xsol);

    rsnr2 = 20*log10(norm(image(:))/norm(image(:)-bp_y(:)));
    ssimval2= ssim(img1,bp_y);

    figure(i), subplot(1,3,1), imagesc(image), axis image,  colorbar, colormap gray
    title('Original image');
    
    figure(i), subplot(1,3,2), imagesc(bp_y), axis image,  colorbar, colormap gray
    title({'image number:',['Back project image. RSNR=',num2str(rsnr2,'%.1f'),' dB.'],['SSIM=',+num2str(ssimval2)]});

    figure(i), subplot(1,3,3), imagesc(xsol), axis image,  colorbar, colormap gray
    title({'image number:',['Reconstructed image in the noisy case. RSNR=',num2str(rsnr,'%.1f'),' dB.'],['SSIM=',+num2str(ssimval)]});
    

    RSNR(i) = rsnr ; 
    SSIM(i) = ssimval ;
    TIME(i) = toc;


    disp('=====================================================================================================================================================================================')
    disp('=====================================================================================================================================================================================\n\n')  

end

averageSNR = mean(RSNR);
averageSSIM = mean(SSIM);
averageTIME = mean(TIME);

%calculate the standard deviation vlue
standSNR = std(RSNR);
standSSIM = std(SSIM);
standTIME = std(TIME);


disp(RSNR);
disp(SSIM);
disp(TIME);
fprintf("the average value of SNR is: %f \n",averageSNR);
fprintf("the average value of SSIM is: %f \n",averageSSIM);
fprintf("the average value of reconstruction time is: %f \n",averageTIME);
fprintf("the standard deciation value of SNR is: %f \n",standSNR);
fprintf("the standard deciation value of SSIM is: %f \n",standSSIM);
fprintf("the standard deciation value of reconstruction time is: %f \n",standTIME);


%% Auxilary functions
% It can be use to generate noisy imageDatastore

function dataOut = addNoise(data)
dataOut = cell(size(data));
    for col = 1:size(data,2)
        for idx = 1:size(data,1)
            temp = data(idx,col);
            sigma = 0.02;
            temp = temp + sigma*randn(size(temp));
            dataOut{idx,col} = temp;
        end
    end
end
