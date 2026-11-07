function res=metricPWW(im1,im2,fim)

% function res=metricPWW(im1,im2,fim)
%
% This function implements Pei-Wei Wang's algorithms for fusion metric.
% im1, im2 -- input images;
% fim      -- fused image;
% res      -- metric value;
%
% IMPORTANT: The size of the images need to be 2X.Simoncelli's toolbox is
% used, so please note a Matlab version 7.0 or previous is preferred.
%
% Z. Liu [July 2009]
%

% Ref: A novel image fusion metric based on multi-scale analysis, pp.965-968, ICSP 2008.
% by Peng-Wei Wang et al.
%

% Haar wavelet is used for this implementation.

%% pre-processing
im1=double(im1);
im2=double(im2);
fim=double(fim);

% some constant values
N=2;    % decomposition level 
alpha1=2/3; alpha2=1/3;

%% wavelet decomposition

[C1,S1]=buildWpyr(im1,N,'haar');
[C2,S2]=buildWpyr(im2,N,'haar');
[Cf,Sf]=buildWpyr(fim,N,'haar');

%% caculate the metric
% use pyrBand.m and pyrlow.m

for i=1:N
    % first input and fused result
    bandNum=(i-1)*3;
    
    H1=pyrBand(C1,S1,bandNum+1); 
    V1=pyrBand(C1,S1,bandNum+2);
    D1=pyrBand(C1,S1,bandNum+3);

    H2=pyrBand(C2,S2,bandNum+1); 
    V2=pyrBand(C2,S2,bandNum+2);
    D2=pyrBand(C2,S2,bandNum+3);

    Hf=pyrBand(Cf,Sf,bandNum+1); 
    Vf=pyrBand(Cf,Sf,bandNum+2);
    Df=pyrBand(Cf,Sf,bandNum+3);
    
    W1=H1.*H1+V1.*V1+D1.*D1;
    W2=H2.*H2+V2.*V2+D2.*D2;
    
    EP1=exp(-abs(H1-Hf))+exp(-abs(V1-Vf))+exp(-abs(D1-Df));
    EP2=exp(-abs(H2-Hf))+exp(-abs(V2-Vf))+exp(-abs(D2-Df));    
    
    
    Q(i)=sum(sum(EP1.*W1+EP2.*W2))/sum(sum(W1+W2));
    
end

Qep=(Q(1).^alpha1)*(Q(2).^alpha2);
res=Qep;
% [PYR, INDICES] = buildWpyr(IM, HEIGHT, FILT, EDGES)
%
% Construct a separable orthonormal QMF/wavelet pyramid on matrix (or vector) IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(IM,FILT).  You can also specify 'auto' to use this value.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Filter can be of even or odd length, but should be symmetric. 
% Default = 'qmf9'.  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.

% Eero Simoncelli, 6/96, based on original lisp code from 1987.

