function [xsol, bp_y, t, temp] = PnP(deN_Model,image,options,reCoef, flag)
   

% Optional input arguments.
if ~isfield(options, 'rel_tol'), options.rel_tol = 1e-4; end
if ~isfield(options, 'rel_tol2'), options.rel_tol2 = 1e-4; end
if ~isfield(options, 'max_iter'), options.max_iter = 200; end
if ~isfield(options, 'delta'), options.delta = 1; end


% maxValue = double(max(max(image)));
% image = double(image)/maxValue;
% myRange = getrangefromclass(image(1));
% newMax = myRange(2);
% newMin = myRange(1);
% image = (image - min(image(:)))*(newMax - newMin)/(max(image(:)) - min(image(:))) + newMin;

ft = 1.4;             % Subsampling rate 
Nx = size(image,1);
Ny = size(image,2); 
super_res=0;                                        % super resolution: to be set to false (0)
seed = mod(flag,5); 

num_meas = floor(Nx/ft);
M = num_meas*Ny;    % Total number of measurements

[A, At, Gw] = generate_data_single(Nx,Ny,ft,super_res,seed);       % Generate the matrices A, At and Gw

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

% maxValue2 = double(max(max(bp_y)));
% bp_y = double(bp_y)/maxValue2;

rsnr2 = 20*log10(norm(image(:))/norm(image(:)-bp_y(:)));
ssimval2= ssim(image,bp_y);
if ssimval2 > 0.035
    options.delta = options.delta*(0.035/ssimval2);   
    options.max_iter = 15;
    if ssimval2 > 0.07
     options.max_iter = 30;
    end

end

% y1 = bp_y;
% nitm=100;
% prec = 1e-4;
% for nit = 1:nitm
%     x = y1;
%     y1 = Phit(x);
%     y1 = Phi(y1);
%     z = x(:)'*y1(:)/norm(x,'fro')^2;
%     % fprintf(1,'iteration %d norm = %g\n',nit,z);
%     if nit ~=1 & norm(z-zold,'fro') < prec
%         fprintf('power iteration converged at iteration %d\n',nit);
%         fprintf('Lipschitz constant = %d \n',sqrt(z));
%         break
%     end
%     zold = z;
%     y1 = y1/norm(y1,'fro');
% end

% options.delta = 1/sqrt(z);


noise = sigma/sqrt(2)*(randn(Nx,Ny) + 1i*randn(Nx,Ny));
%noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);

delta = 1/(max(svd(noise)));

% if delta < options.delta
%     options.delta = 0.95*delta*reCoef; 
% end

options.delta = options.delta*reCoef; 

epsilon = sigma*sqrt(M + 2*sqrt(M));

%proximity operator
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling

%% Initializations.
%Dual variable.
v = zeros(size(y));

%Initial solution (all zero solution)
%Initial residual/intermediate variable
s = -y;

%Initial l2 projection
n = sc(v-s) ;

%Creating the initial solution variable with all zeros
xsol = zeros(size(real(Phi(s))));

%% Main loop. 

g = @(z) predict(deN_Model,real(z));
cal_s = @(x)Phit(x)-y;


for t = 1: options.max_iter
    xsol_old = xsol;

    a = s+n-v;   

    z = xsol-options.delta.*real(Phi(a));
    xsol = double(g(z));

%     maxValue = double(max(max(xsol)));
%     xsol = double(xsol)/maxValue;

%     myRange = getrangefromclass(xsol(1));
%     newMax = myRange(2);
%     newMin = myRange(1);
%     xsol = (xsol - min(xsol(:)))*(newMax - newMin)/(max(xsol(:)) - min(xsol(:))) + newMin;

    s = cal_s(xsol);
    n = sc(v - s);
    v = v - (s+n);
    
   
    testflag = 100;
    flag = isnan(xsol(1,1:testflag));

    if any(flag)  
        xsol = xsol_old;
        break;
    end

    if xsol(1,1:testflag) == zeros(1,testflag)        
        temp = 3;
        disp('Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Picture is damaged, recalculate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        break;
    end
    
end

temp = 1;

myRange = getrangefromclass(xsol(1));
newMax = myRange(2);
newMin = myRange(1);
xsol = (xsol - min(xsol(:)))*(newMax - newMin)/(max(xsol(:)) - min(xsol(:))) + newMin;

rsnr_sol = 20*log10(norm(image(:))/norm(image(:)-xsol(:)));

if rsnr_sol < 0 || isnan(rsnr_sol)
    temp = 2;
end




% if reCoef
%     fprintf('It went through %d iterations. \nAnd the final xsol is',t);
%     disp(xsol(1,1:12));
% end
