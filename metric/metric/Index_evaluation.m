function res=Index_evaluation(I1,I2,F,k)
    I1=double(I1);
    I2=double(I2);
    F=double(F);
    Q(1,1)=metricMI(I1,I2,F,1);        %Q_NMI
    xlswrite('D:\Evaluation.xlsx',Q(1,1),4,strcat('A',num2str(k+1)))
    Q(1,2)=metric_G(I1,I2,F);          %第一种Q_G
    xlswrite('D:\Evaluation.xlsx',Q(1,2),4,strcat('B',num2str(k+1)))
    Q(1,3)=metricYang(I1,I2,F);        %Q_Y
    xlswrite('D:\Evaluation.xlsx',Q(1,3),4,strcat('C',num2str(k+1)))
    Q(1,4)=metricChenBlum(I1,I2,F);    %Q_CB
    xlswrite('D:\Evaluation.xlsx',Q(1,4),4,strcat('D',num2str(k+1)))
    Q(1,5)=metricMI(I1,I2,F,3);        %Q_TE
    xlswrite('D:\Evaluation.xlsx',Q(1,5),4,strcat('E',num2str(k+1)))
    Q(1,6)=metricWang(I1,I2,F);        %Q_NCIE
    xlswrite('D:\Evaluation.xlsx',Q(1,6),4,strcat('F',num2str(k+1)))
    Q(1,7)=metricXydeas(I1,I2,F);      %第二种Q_G
    xlswrite('D:\Evaluation.xlsx',Q(1,7),4,strcat('G',num2str(k+1)))
    Q(1,8)=metricZheng(I1,I2,F);       %Q_SF
    xlswrite('D:\Evaluation.xlsx',Q(1,8),4,strcat('H',num2str(k+1)))
    Q(1,9)=metricZhao(I1,I2,F);        %Q_P
    xlswrite('D:\Evaluation.xlsx',Q(1,9),4,strcat('I',num2str(k+1)))
    Q(1,10)=metricPeilla(I1,I2,F,1);   %Q_Piella
    xlswrite('D:\Evaluation.xlsx',Q(1,10),4,strcat('J',num2str(k+1)))
    Q(1,11)=metricCvejic(I1,I2,F,2);   %Q_C
    xlswrite('D:\Evaluation.xlsx',Q(1,11),4,strcat('K',num2str(k+1)))
    Q(1,12)=metricChen(I1,I2,F);       %Q_CV
    xlswrite('D:\Evaluation.xlsx',Q(1,12),4,strcat('L',num2str(k+1)))
    Q(1,13)=analysis_Qabf(I1,I2,F);       %Q_abf
    xlswrite('D:\Evaluation.xlsx',Q(1,13),4,strcat('M',num2str(k+1)))
    Q(1,14)=metricsAvg_gradient(I1,I2,F);       %AVG
    xlswrite('D:\Evaluation.xlsx',Q(1,14),4,strcat('N',num2str(k+1)))
    Q(1,15)=VIFF_Public(I1,I2,F);       %VIFF
    xlswrite('D:\Evaluation.xlsx',Q(1,15),4,strcat('O',num2str(k+1)))
    Q(1,16)=analysis_SCD(I1,I2,F);      %Q_SCD(差异相关性总和)
    xlswrite('D:\Evaluation.xlsx',Q(1,16),4,strcat('P',num2str(k+1)))
    Q(1,17)=analysis_fmi(I1,I2,F);      %FMI_pixel
    xlswrite('D:\Evaluation.xlsx',Q(1,17),4,strcat('Q',num2str(k+1)))
    Q(1,18)=analysis_fmi(I1,I2,F,'dct');      %FMI_dct
    xlswrite('D:\Evaluation.xlsx',Q(1,18),4,strcat('R',num2str(k+1)))
    Q(1,19)=analysis_fmi(I1,I2,F,'wavelet');      %FMI_w
    xlswrite('D:\Evaluation.xlsx',Q(1,19),4,strcat('S',num2str(k+1)))
    res=Q;
end