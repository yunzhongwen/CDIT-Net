clear all;
close all;
clc;

%% 图像集配置
dataset_list = {
    % 名称             source1_path                                      source2_path                                      result_path
   % 'MFFW',           'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\MFFW\sourceA',      'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\MFFW\sourceB',        'H:\qbm\DuiBi  ShiYan\Text-MFF\Datasets-test\mffw';
    'Grayscale',      'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\Grayscale\sourceA',  'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\Grayscale\sourceB',  'H:\qbm\compare\Mask-Difuser(2026)\Grayscale\Grayscale';       
   % 'Lytro',          'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\Lytro\sourceA',      'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\Lytro\sourceB',      'H:\qbm\DuiBi  ShiYan\Text-MFF\Datasets-test\Lytro';
    % 'MFI-WHU',        'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\MFI-WHU\sourceA',    'C:\Users\Administrator\Desktop\mynetwork (2)\Ranni\Datasets\Eval\MFI-WHU\sourceB',    'H:\qbm\DuiBi  ShiYan\Text-MFF\Datasets-test\MFI';
    };

% 指标标题：重命名避免和Matlab内置title函数冲突
metric_titles = {'Q_NMI','Q_sG','Q_Y(SSIM)','Q_CB','Q_TE(EN)','Q_NCIE','Q_G','Q_SF(空间频率)',...
    'Q_P','Q_Piella(Q_S)','Q_C','Q_CV','Q_SCD','Q_ABF','AVG','VIFF', ...
    'FMI_pixel','FMI_dct','FMI_w','PSNR','MSE'};

%% 遍历每个数据集
for i = 1:size(dataset_list, 1)
    dataset_name = dataset_list{i, 1};
    source1_path = dataset_list{i, 2};
    source2_path = dataset_list{i, 3};
    result_path = dataset_list{i, 4};

    % 确保结果保存路径存在（不存在则创建）
    if ~exist(result_path, 'dir')
        mkdir(result_path);
        fprintf('创建结果目录：%s\n', result_path);
    end

    fprintf('\n正在处理数据集：%s\n', dataset_name);

    % 读取图像列表（只匹配.jpg格式）
    img_path1_list = dir(fullfile(source1_path, '*.jpg'));
    img_path2_list = dir(fullfile(source2_path, '*.jpg'));
    img_path3_list = dir(fullfile(result_path, '*.jpg'));
    img_num = length(img_path2_list);

    if img_num > 0
        data = []; % 存储每张图的指标结果
        for j = 1:img_num
            try
                % 读取图像（处理文件不存在的情况）
                img1_fullpath = fullfile(source1_path, img_path1_list(j).name);
                img2_fullpath = fullfile(source2_path, img_path2_list(j).name);
                f_fullpath = fullfile(result_path, img_path3_list(j).name);
                
                if ~exist(img1_fullpath, 'file') || ~exist(img2_fullpath, 'file') || ~exist(f_fullpath, 'file')
                    error(sprintf('图像文件缺失：%s / %s / %s', img1_fullpath, img2_fullpath, f_fullpath));
                end

                I1 = imread(img1_fullpath);
                I2 = imread(img2_fullpath);
                F  = imread(f_fullpath);

                % 转灰度图
                if size(I1, 3) == 3, I1 = rgb2gray(I1); end
                if size(I2, 3) == 3, I2 = rgb2gray(I2); end
                if size(F,  3) == 3, F  = rgb2gray(F);  end

                % 裁剪统一尺寸
                [I1, I2, F] = align_image_sizes(I1, I2, F);

                % double化后计算指标
                E_index = Index_evaluation2(double(I1), double(I2), double(F));
                fprintf('第 %d 张图计算完成：%s\n', j, img_path2_list(j).name);
                data = [data; E_index];
            catch ME
                fprintf('第 %d 张图处理失败: %s\n', j, ME.message);
                % 失败时填充NaN，保证维度一致
                data = [data; nan(1, length(metric_titles))];
            end
        end

        % ========== 关键：先定义result_cell，再调用writecell ==========
        % 计算所有图像指标的均值（忽略NaN）
        mean_row = mean(data, 1, 'omitnan');
        data_with_mean = [data; mean_row];
        
        % 拼接标题和数据为元胞数组（核心：正确定义result_cell）
        result_cell = [metric_titles; num2cell(data_with_mean)];

        % 保存结果：替换xlswrite为writecell（适配元胞数组）
        excel_file = fullfile(result_path, [dataset_name, '.xlsx']);
        writecell(result_cell, excel_file, 'Sheet', 1, 'Range', 'A1');
        fprintf('结果已保存至：%s\n', excel_file);
    else
        fprintf('数据集 %s 中未找到.jpg图像，跳过。\n', dataset_name);
    end
end

%% ============== 工具函数 ==============
function [img1, img2, img3] = align_image_sizes(img1, img2, img3)
    % 统一三张图像的尺寸（取最小高宽）
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
    % 计算21项图像融合评价指标
    Q = zeros(1, 21); % 初始化指标数组，避免维度错误
    Q(1)  = metricMI(I1,I2,F,1);        % Q_NMI
    Q(2)  = metric_G(I1,I2,F);          % Q_sG
    Q(3)  = metricYang(I1,I2,F);        % Q_SSIM
    Q(4)  = metricChenBlum(I1,I2,F);    % Q_CB
    Q(5)  = metricMI(I1,I2,F,3);        % Q_TE
    Q(6)  = metricWang(I1,I2,F);        % Q_NCIE
    Q(7)  = metricXydeas(I1,I2,F);      % Q_G
    Q(8)  = metricZheng(I1,I2,F);       % Q_SF
    Q(9)  = metricZhao(I1,I2,F);        % Q_P
    Q(10) = metricPeilla(I1,I2,F,1);    % Q_Piella
    Q(11) = metricCvejic(I1,I2,F,2);    % Q_C
    Q(12) = metricChen(I1,I2,F);        % Q_CV
    Q(13) = analysis_SCD(I1,I2,F);      % Q_SCD
    Q(14) = analysis_Qabf(I1,I2,F);     % Q_ABF
    Q(15) = metricsAvg_gradient(I1,I2,F);% AVG
    Q(16) = VIFF_Public(I1,I2,F);       % VIFF
    Q(17) = analysis_fmi(I1,I2,F);          % FMI_pixel
    Q(18) = analysis_fmi(I1,I2,F,'dct');    % FMI_dct
    Q(19) = analysis_fmi(I1,I2,F,'wavelet');% FMI_wavelet
    Q(20) = PSNR_evaluation(I1, I2, F);     % PSNR
    Q(21) = MSE_evaluation(I1, I2, F);      % MSE
    res = Q;
end