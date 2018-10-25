name_list_pca = {'le', 'lbp_hd', 'cnn_v7_p9'};

% load('result_lfw_pca_le.mat')
outDir = '../result/';
flag = 'all_pca_';
p = 3;

line_style = {'-', '--', '-+', '-o', '-.'};

figure;
for i=1:1:p

    file_name = strcat('result_lfw_pca_', name_list_pca(1, i), '.mat');
    file_name = char(file_name);
    load(file_name);
    
    semilogx(meanVeriFAR*100, fusedVR, char(line_style(1, i)), 'LineWidth', 1);
    
    hold on;

end

legend('LE', 'High Dim LBP', 'CNN Model', 'Location', 'best');
xlim([0,100]); ylim([0,100]); 
grid on;
xlabel('False Accept Rate (%)');
ylabel('Verification Rate (%)');
title('Face Verification ROC Curve');
figurename = strcat(outDir, flag, 'verificationROC'); 
print(gcf, figurename,'-dpdf');
% print(gcf, figurename,'-dpng');

figure;
for i=1:1:p
    file_name = strcat('result_lfw_pca_', name_list_pca(1, i), '.mat');
    file_name = char(file_name);
    load(file_name);
    semilogx(meanOsiFAR*100, fusedDIR(rankIndex,:), char(line_style(1, i)), 'LineWidth', 1);
    hold on;
end

legend('LE', 'High Dim LBP', 'CNN Model', 'Location', 'best');
xlim([0,100]); ylim([0,50]); 
grid on;
xlabel('False Accept Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set Identification ROC Curve at Rank %d', reportRank));
figurename = strcat(outDir, flag, 'rank1'); 
print(gcf, figurename,'-dpdf');
% print(gcf, figurename,'-dpng');

%% Plot the face verification ROC curve.

%semilogx(meanVeriFAR * 100, fusedVR, 'LineWidth', 2);


%% Plot the open-set face identification ROC curve at the report rank.

