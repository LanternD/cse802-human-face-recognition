function [] = feature_csv_to_mat()

function_flag = 0;
if function_flag == 0
    filename = './lfw_feature_matrix.csv';
    Descriptors = csvread(filename);
    %Descriptors
    save('./BLUFR/data/lfw_cnn_v7_p9.mat', 'Descriptors')
end

if function_flag == 1
    
    feaFile = './BLUFR/data/high_dim_lbp_lfw.mat';
    load(feaFile, 'high_dim_LBP_LFW');
    Descriptors = high_dim_LBP_LFW;
    save('./BLUFR/data/lfw_lbp_hd.mat', 'Descriptors')
end


if function_flag == 2
    feaFile = './BLUFR/data/le_lfw.mat';
    load(feaFile, 'le_lfw');
    Descriptors = le_lfw;
    save('./BLUFR/data/lfw_le.mat', 'Descriptors')    
end
end
