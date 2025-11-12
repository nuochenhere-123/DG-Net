function [final_feature] = NUIQ_feature_extraction(I,group,num_ingroup)
%% Feature extraction
D_AGGD = []; A_AGGD = []; S_AGGD = []; Ax_AGGD = []; Ay_AGGD = [];
moment = []; O3_LBP = []; GM_LBP = []; GO_LBP = [];
shifts = [ 0 1; 1 0; 1 1; -1 1];
%% Color Space Conversion
I_r = double(I(:,:,1));
I_g = double(I(:,:,2));
I_b = double(I(:,:,3));
O1 = (I_r-I_g)/sqrt(2);
O2 = (I_r+I_g-2*I_b)/sqrt(6);
O3 = (I_r+I_g+I_b)/sqrt(3);
%% moment statistics
vector = ColorMoments(I);
moment = [moment;vector];
moment = abs(moment);

%% opponent difference map D 
D = abs(O1-O2);
% AGGD fit
k_num = 0.01;
C1 = (k_num*255)^2;
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
mu            = filter2(window, D, 'same');
mu_sq         = mu.*mu;
sigma         = sqrt(abs(filter2(window, D.*D, 'same') - mu_sq));
structdis     = (D-mu)./(sigma+C1);
structdis = mapminmax(structdis(:),0,1);
[alpha, leftstd, rightstd]  = estimateaggdparam(structdis);
const                    =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
meanparam                =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
D_AGGD = [D_AGGD;alpha,meanparam,leftstd^2,rightstd^2];

%% saturation map S
S = sqrt(O1.*O1+O2.*O2);
% AGGD fit
shifted_s = circshift(S, shifts(1,:));
diff_s = S - shifted_s;
diff_s = diff_s(2:end-1,2:end-1); diff_s = diff_s(:);
[alpha, leftstd, rightstd]  = estimateaggdparam(diff_s);
const                    =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
meanparam                =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
S_AGGD = [S_AGGD;alpha meanparam leftstd^2 rightstd^2];
clear diff_s paramEsts                   

%% opponent angle map A
A = atan(O1./O2);
% AGGD fit
shifted_h = circshift(A, shifts(1,:));
diff_h = A - shifted_h;
diff_h = diff_h(2:end-1,2:end-1); diff_h = diff_h(:);
[alpha, leftstd, rightstd]  = estimateaggdparam(diff_h);
const                    =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
meanparam                =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
A_AGGD = [A_AGGD;alpha meanparam leftstd^2 rightstd^2];
clear diff_h paramEsts

%% opponent derivation angle maps Ax and Ay
[O1_x,O1_y]=gradient(O1);
[O2_x,O2_y]=gradient(O2);
Ax = atan(O1_x./O2_x);
shifted_oax = circshift(Ax, shifts(1,:));
diff_oax = Ax - shifted_oax;
diff_oax = diff_oax(2:end-1,2:end-1); diff_oax = diff_oax(:);
[alpha, leftstd, rightstd]  = estimateaggdparam(diff_oax);
const                    =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
meanparam                =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
Ax_AGGD = [Ax_AGGD;alpha meanparam leftstd^2 rightstd^2];
clear diff_oax paramEsts

Ay = atan(O1_y./O2_y);
shifted_oay = circshift(Ay, shifts(1,:));
diff_oay = Ay - shifted_oay;
diff_oay = diff_oay(2:end-1,2:end-1); diff_oay = diff_oay(:);
[alpha, leftstd, rightstd]  = estimateaggdparam(diff_oay);
const                    =(sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
meanparam                =(rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
Ay_AGGD = [Ay_AGGD;alpha meanparam leftstd^2 rightstd^2];
clear diff_oay paramEsts
%% Mean-Substracted Contrast Normalized Map -> LBP
gray_I = double(rgb2gray(I));

scalenum = 2;
k_num = 0.01;
C1 = (k_num*255)^2;

window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));

mu            = filter2(window, gray_I, 'same');
mu_sq         = mu.*mu;
sigma         = sqrt(abs(filter2(window, gray_I.*gray_I, 'same') - mu_sq));
structdis     = (gray_I-mu)./(sigma+C1);
mapping=getmapping(8,'riu2');
LBP_feat_O3=lbp(structdis,1,8,mapping,'nh');
O3_LBP = [O3_LBP;LBP_feat_O3];

%% Gradient Magnitude and Orientation Maps -> LBP
% Gaussian filter
sigma=0.5;
x=[-1 0 1;-1 0 1;-1 0 1];
y=[-1 -1 -1;0 0 0;1 1 1];
gdx = -(x/(2*pi*sigma^4)).*exp(-(x.*x + y.*y)./(2*sigma^2));
gdy = -(y/(2*pi*sigma^4)).*exp(-(x.*x + y.*y)./(2*sigma^2));
s=(1/9).*[1,1,1;1,1,1;1,1,1];
im_dy = imfilter(double(gray_I), gdy, 'conv');
im_dx = imfilter(double(gray_I), gdx','conv');

% Gradient Magnitude and Orientation Maps
GM = sqrt(im_dx.^2 + im_dy.^2);
GM = (GM-min(GM(:)))./(max(GM(:))-min(GM(:)))*255;
GM = uint8(GM); % double->int
mapping=getmapping(8,'riu2');
LBP_feat_GM=lbp(GM,1,8,mapping,'nh');
GM_LBP = [GM_LBP;LBP_feat_GM];

GO = atan2(im_dy,im_dx); 
GO=round(GO./(pi/4)).*(pi/4);%对img_grad_or逐点除以pi/4,利用round取整，再逐点乘以pi/4完成量化
mapping=getmapping(8,'riu2');
LBP_feat_GO=lbp(GO,1,8,mapping,'nh');
GO_LBP = [GO_LBP;LBP_feat_GO];

final_feature = [D_AGGD,S_AGGD,A_AGGD,Ax_AGGD,Ay_AGGD,moment,O3_LBP,GM_LBP,GO_LBP];
if any(isnan(final_feature(:)))
    final_feature(isnan(final_feature)) = 0; % 将NaN值替换为0
end
end

