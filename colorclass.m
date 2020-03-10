I=imread('b (4).jpg');
% Extract RGB Channel
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);
% Extract Statistical features
% 1] MEAN
meanR=mean2(R);
meanG=mean2(G);
meanB=mean2(B);
% 2] Standard Deviation
stdR=std2(R);
stdG=std2(G);
stdB=std2(B);
A=[meanR meanG meanB stdR stdG stdB];