clear all;
close all;
clc;


for k=1:9
input_adressA=strcat('F:\A_Matte_MFIFGAN\MFIF-GAN-main\dataset\\test_dataset\Lytro\img_A\','lytro-0',num2str(k),'-A.jpg');
input_adressB=strcat('F:\A_Matte_MFIFGAN\MFIF-GAN-main\dataset\\test_dataset\Lytro\img_B\','lytro-0',num2str(k),'-B.jpg');
input_adressF=strcat('D:\实验数据\PCANet-master\lytro\','lytro_0',num2str(k),'_FusionDN.jpg');
I1=imread(input_adressA);
I2=imread(input_adressB);
F=imread(input_adressF);

[H,W,D]=size(I1);
if D==3
    channel=3;
    I1=rgb2gray(I1);
    I2=rgb2gray(I2);
    F=rgb2gray(F);
else
    channel=1;
end

I1=double(I1);
I2=double(I2);
F=double(F);

E_index=Index_evaluation(I1,I2,F,k);
fprintf('%.4f\n',E_index)
% E=Fusion_Assess(I1,I2,F);
end

for k=10:20
input_adressA=strcat('F:\A_Matte_MFIFGAN\MFIF-GAN-main\dataset\\test_dataset\Lytro\img_A\','lytro-',num2str(k),'-A.jpg');
input_adressB=strcat('F:\A_Matte_MFIFGAN\MFIF-GAN-main\dataset\\test_dataset\Lytro\img_B\','lytro-',num2str(k),'-B.jpg');
input_adressF=strcat('D:\实验数据\PCANet-master\lytro\','lytro_',num2str(k),'_FusionDN.jpg');
I1=imread(input_adressA);
I2=imread(input_adressB);
F=imread(input_adressF);

[H,W,D]=size(I1);
if D==3
    channel=3;
    I1=rgb2gray(I1);
    I2=rgb2gray(I2);
    F=rgb2gray(F);
else
    channel=1;
end

I1=double(I1);
I2=double(I2);
F=double(F);

E_index=Index_evaluation(I1,I2,F,k);
fprintf('%.4f\n',E_index)
% E=Fusion_Assess(I1,I2,F);
end