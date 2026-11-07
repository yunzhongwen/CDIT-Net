function res=Index_evaluation2(I1,I2,F,k)
    I1=double(I1);
    I2=double(I2);
    F=double(F);
    
    
    Q(1,1)=metricMI(I1,I2,F,1);        %Q_NMI
    Q(1,2)=metric_G(I1,I2,F);          %Q_sG
    Q(1,3)=metricYang(I1,I2,F);        %Q_SSIM(结构相似性)
    Q(1,4)=metricChenBlum(I1,I2,F);    %Q_CB
    Q(1,5)=metricMI(I1,I2,F,3);        %Q_TE(交叉熵)
    Q(1,6)=metricWang(I1,I2,F);        %Q_NCIE
    Q(1,7)=metricXydeas(I1,I2,F);      %Q_G(基于梯度)
    Q(1,8)=metricZheng(I1,I2,F);       %Q_SF(Spatial Frequency)
    Q(1,9)=metricZhao(I1,I2,F);        %Q_P(相位一致性)
    Q(1,10)=metricPeilla(I1,I2,F,1);   %Q_Piella/Q_S
    Q(1,11)=metricCvejic(I1,I2,F,2);   %Q_C
    Q(1,12)=metricChen(I1,I2,F);       %Q_CV
    Q(1,13)=analysis_SCD(I1,I2,F);      %Q_SCD(差异相关性总和)
    Q(1,14)=analysis_Qabf(I1,I2,F);     %Q_ABF(C.S.Xydeas)
    Q(1,15)=metricsAvg_gradient(F);    %AVG(平均梯度)
    Q(1,16)=VIFF_Public(I1,I2,F);      %VIFF
    
 
    res=Q;
end