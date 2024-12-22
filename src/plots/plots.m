
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        Plotting some results       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all, clear all, clc 


% Load tables
results1 = readtable('results/report_finetuning_configuration_1.csv');
results2 = readtable('results/report_finetuning_configuration_2.csv');


% Sort by accuracy
results1 = sortrows(results1, 5, 'descend');
results2 = sortrows(results2, 4, 'descend');

sortedLabels1 = arrayfun(@(hu, tf, lr, m) sprintf('HU: %d, TF: %s, LR: %.3f M:%.1f', hu, tf{:}, lr, m), ...
                        results1.(1), results1.(2), results1.(3), results1.(4), 'UniformOutput', false);
sortedLabels2 = arrayfun(@(hu, tf, lr) sprintf('HU: %d, TF: %s, LR: %.3f', hu, tf{:}, lr), ...
                        results2.(1), results2.(2), results2.(3), 'UniformOutput', false);


% Bar plot Configuration 1
figure (1);
set(gcf,'position',[0,0,1400,800])

bar(results1.(5), 'FaceColor', [0.2, 0.6, 0.8]); 
xticks(1:height(results1)); 
xticklabels(sortedLabels1);  
xtickangle(45); 
ylabel('Accuracy (%)'); 
title('Fine-tuning Configuration 1'); 
grid on; % Add grid for better visualization

disp('Sorted Sets of Parameters by Accuracy:');
disp(results1);

saveas(gcf, 'results/plot_finetuning_configuration_1.png'); % Saves as a PNG file


% Bar plot Configuration 2
figure (2);
set(gcf,'position',[0,0,1400,800])

bar(results2.(4), 'FaceColor', [0.2, 0.6, 0.8]); 
xticks(1:height(results2)); 
xticklabels(sortedLabels2);  
xtickangle(45); 
ylabel('Accuracy (%)'); 
title('Fine-tuning Configuration 2'); 
grid on; % Add grid for better visualization

disp('Sorted Sets of Parameters by Accuracy:');
disp(results2);

saveas(gcf, 'results/plot_finetuning_configuration_2.png'); % Saves as a PNG file