
% Tetsing required preprocessing steps
load('datasets/caltech101_silhouettes_28.mat')
X_normalized = X / 255; 
Y = Y(:)'
numClasses = length(classnames);
Y_onehot = full(ind2vec(Y + 1)); 