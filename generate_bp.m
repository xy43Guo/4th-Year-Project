close all;
clear;
clc;

addpath('../utils/');
addpath('../utils/lib/');
addpath('..');
run('utils/lib/irt/setup.m');

FilePath = 'dataset_single/testingset3/';%qiang
FilePath3 = 'bpimage3/';

read_files = dir(fullfile(FilePath, '*.fits'));
img_num = length(read_files);


for i = 1: img_num
    disp(i)

    filename = read_files(i).name;

    image_name=read_files(i).name;
    image = fitsread(strcat(FilePath,image_name));

    ft = 1.4;             % Subsampling rate 
    Nx = size(image,1);
    Ny = size(image,2); 
    super_res=0;                                        % super resolution: to be set to false (0)
    seed=mod(img_num,5); 
    
    num_meas = floor(Nx/ft);
    M = num_meas*Ny;    % Total number of measurements
    
    [A, At, Gw] = generate_data_single01(Nx,Ny,ft,super_res,seed);       % Generate the matrices A, At and Gw
    
    Phit = @(x) HS_forward_operator_matrix(x,Gw,A)/sqrt(Nx*Ny);                  % Forward (measurement) operator
    Phi = @(y) real(HS_adjoint_operator_matrix(y,Gw,At,Nx,Ny))/sqrt(Nx*Ny);             % Adjoint operator
    
    y = Phit(image);
    isnr = 30;
    sigma = norm(y(:))/sqrt(numel(y))*10^(-isnr/20);
    y = y+(sigma*randn(size(y))+1i*sigma*randn(size(y)))/sqrt(2);
    
    bp_y = real(Phi(y));
    myRange = getrangefromclass(bp_y(1));
    newMax = myRange(2);
    newMin = myRange(1);
    bp_y = (bp_y - min(bp_y(:)))*(newMax - newMin)/(max(bp_y(:)) - min(bp_y(:))) + newMin;

    filename(end-3:end) = 'fits';               
    write_filename = [FilePath3 filename];  % fullfile name to write
    
    fitswrite(bp_y,write_filename);
end