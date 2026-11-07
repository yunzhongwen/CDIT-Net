clear all;
close all;
clc;

%% 图像集配置
dataset_list = {
    % 名称             source1_path                                      source2_path                                      result_path
    'MFFW',           'D:\DENG Minyu\Datasets\Eval\MFFW\sourceA\',      'D:\DENG Minyu\Datasets\Eval\MFFW\sourceB\',        'D:\DENG Minyu\Results\最终1.0\消融实验\backbone\Results\MFFW\';
   % 'Grayscale',      'D:\DENG Minyu\Datasets\Eval\Grayscale\sourceA\',  'D:\DENG Minyu\Datasets\Eval\Grayscale\sourceB\',  'D:\DENG Minyu\Results\最终1.0\消融实验\backbone\Results\Grayscale\';
    'Lytro',          'D:\DENG Minyu\Datasets\Eval\Lytro\sourceA\',      'D:\DENG Minyu\Datasets\Eval\Lytro\sourceB\',      'D:\DENG Minyu\Results\最终1.0\消融实验\backbone\Results\Lytro\';
    'MFI-WHU',        'D:\DENG Minyu\Datasets\Eval\MFI-WHU\sourceA\',    'D:\DENG Minyu\Datasets\Eval\MFI-WHU\sourceB\',    'D:\DENG Minyu\Results\最终1.0\消融实验\backbone\Results\MFI-WHU\';
    };

% 指标标题
title = {'Q_NMI','Q_sG','Q_Y(SSIM)','Q_CB','Q_TE(EN)','Q_NCIE','Q_G','Q_SF(空间频率)',...
    'Q_P','Q_Piella(Q_S)','Q_C','Q_CV','Q_SCD','Q_ABF','AVG','VIFF', ...
    'FMI_pixel','FMI_dct','FMI_w','PSNR','MSE'};

%% 遍历每个数据集
for i = 1:size(dataset_list, 1)
    dataset_name = dataset_list{i, 1};
    source1_path = dataset_list{i, 2};
    source2_path = dataset_list{i, 3};
    result_path = dataset_list{i, 4};

    fprintf('\n正在处理数据集：%s\n', dataset_name);

    % 读取图像列表
    img_path1_list = dir(fullfile(source1_path, '*.jpg'));
    img_path2_list = dir(fullfile(source2_path, '*.jpg'));
    img_path3_list = dir(fullfile(result_path, '*.png'));
    img_num = length(img_path2_list);

    if img_num > 0
        data = [];
        for j = 1:img_num
            try
                I1 = imread(fullfile(source1_path, img_path1_list(j).name));
                I2 = imread(fullfile(source2_path, img_path2_list(j).name));
                F  = imread(fullfile(result_path,  img_path3_list(j).name));

                % 转灰度图
                if size(I1, 3) == 3, I1 = rgb2gray(I1); end
                if size(I2, 3) == 3, I2 = rgb2gray(I2); end
                if size(F,  3) == 3, F  = rgb2gray(F);  end

                % 裁剪统一尺寸
                [I1, I2, F] = align_image_sizes(I1, I2, F);

                % double 化后计算指标
                E_index = Index_evaluation2(double(I1), double(I2), double(F));
                fprintf('第 %d 张图计算完成。\n', j);
                data = [data; E_index];
            catch ME
                fprintf('第 %d 张图处理失败: %s\n', j, ME.message);
                data = [data; nan(1, length(title))];
            end
        end

        % 统计均值
        data = [data; mean(data, 'omitnan')];
        [m, p] = size(data);
        data_cell = mat2cell(data, ones(m,1), ones(p,1));
        result = [title; data_cell];

        % 保存结果
        excel_file = fullfile(result_path, strcat(dataset_name, '.xlsx'));
        xlswrite(excel_file, result);
        fprintf('保存完成：%s\n', excel_file);
    else
        fprintf('数据集 %s 中图像数量为0，跳过。\n', dataset_name);
    end
end


%% ============== 工具函数 ==============
function [img1, img2, img3] = align_image_sizes(img1, img2, img3)
    [h1, w1] = size(img1);
    [h2, w2] = size(img2);
    [h3, w3] = size(img3);

    min_h = min([h1, h2, h3]);
    min_w = min([w1, w2, w3]);

    img1 = img1(1:min_h, 1:min_w);
    img2 = img2(1:min_h, 1:min_w);
    img3 = img3(1:min_h, 1:min_w);
end

function res = Index_evaluation2(I1, I2, F)
    Q(1,1)  = metricMI(I1,I2,F,1);        % Q_NMI
    Q(1,2)  = metric_G(I1,I2,F);          % Q_sG
    Q(1,3)  = metricYang(I1,I2,F);        % Q_SSIM
    Q(1,4)  = metricChenBlum(I1,I2,F);    % Q_CB
    Q(1,5)  = metricMI(I1,I2,F,3);        % Q_TE
    Q(1,6)  = metricWang(I1,I2,F);        % Q_NCIE
    Q(1,7)  = metricXydeas(I1,I2,F);      % Q_G
    Q(1,8)  = metricZheng(I1,I2,F);       % Q_SF
    Q(1,9)  = metricZhao(I1,I2,F);        % Q_P
    Q(1,10) = metricPeilla(I1,I2,F,1);    % Q_Piella
    Q(1,11) = metricCvejic(I1,I2,F,2);    % Q_C
    Q(1,12) = metricChen(I1,I2,F);        % Q_CV
    Q(1,13) = analysis_SCD(I1,I2,F);      % Q_SCD
    Q(1,14) = analysis_Qabf(I1,I2,F);     % Q_ABF
    Q(1,15) = metricsAvg_gradient(I1,I2,F);% AVG
    Q(1,16) = VIFF_Public(I1,I2,F);       % VIFF
    Q(1,17) = analysis_fmi(I1,I2,F);          % FMI_pixel
    Q(1,18) = analysis_fmi(I1,I2,F,'dct');    % FMI_dct
    Q(1,19) = analysis_fmi(I1,I2,F,'wavelet');% FMI_wavelet
    Q(1,20) = PSNR_evaluation(I1, I2, F);     % PSNR
    Q(1,21) = MSE_evaluation(I1, I2, F);      % MSE
    res = Q;
end
