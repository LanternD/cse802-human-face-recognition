%load data;
data=load('data_eliminated.mat');
label=load('labels_eliminated.mat');
X=data.fea;
y=label.class_labels;


%center and scale;
[nrow,ncol]=size(X);
for i=1:ncol
    m=mean(X(:,i));
    sd=std(X(:,i));
    X(:,i)=(X(:,i)-m)/sd;
end

% shape=['o','+','d','*','s','.','v','^']
% %PCA;
% [coeff,score,latent] = pca(X);
% 
% %lda;
% [mappedX_lda, mapping_lda] = lda(X, y, 3);
% 
% %isomap;
% [mappedX_iso, mapping_iso] = isomap(X, 3);

figure();
hold on
   for i=1:nrow
       sym=shape(y(i));
       scatter(score(i,1),score(i,2),40,sym);
   end

 
% figure();
% hold on
% grid on
%    for i=1:nrow
%        sym=shape(y(i));
%        scatter3(score(i,1),score(i,2),score(i,3),40,sym); 
%    end
% 
% 
% 
% figure();
% hold on
%    for i=1:nrow
%        sym=shape(y(i));
%        scatter(mappedX_lda(i,1),mappedX_lda(i,2),40,sym);
%    end
%    
% figure();
% hold on
% grid on
%    for i=1:nrow
%        sym=shape(y(i));
%        scatter3(mappedX_lda(i,1),mappedX_lda(i,2),mappedX_lda(i,3),40,sym);
%    end
% 
% 
%    figure();
% hold on
%    for i=1:nrow
%        sym=shape(y(i));
%        scatter(mappedX_iso(i,1),mappedX_iso(i,2),40,sym);
%    end
%    
% figure();
% hold on
% grid on
%    for i=1:nrow
%        sym=shape(y(i));
%        scatter3(mappedX_iso(i,1),mappedX_iso(i,2),mappedX_iso(i,3),40,sym);
%    end
% 
% 
% 
% 
% 
