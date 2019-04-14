clc
clear all
close all
Data =datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',18000);
T=read(Data);
X=T(:,4:21);
X=X{:,:};
Alpha=0.01;
lamda=0.001;
Corrx = corr(X);
Covx= cov(X);
[U S V] = svd(Covx);
[n m]= size(X);
for w=1:m
    if max(abs(X(:,w)))~=0;
        X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
        
    end
end
alpha=0.5;
K=0;
while (alpha>=0.001)
    K=K+1;
    lamdas(K,:)=sum(max(S(:,1:K)));
    lamdass=sum(max(S));
    alpha=1-lamdas./lamdass;    
end
R=U(:, 1:K)'*(X)';
app_data=U(:,1:K)*R;
error=(1/m)*(sum(app_data-X'));
h=1;
Theta=zeros(m,1);
k=1;
Y=T{:,3}/mean(T{:,3});
E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2); %cost function
while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*app_data*(app_data'*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2);
    
    %Regularization
    Reg(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.0001;
        h=0;
    end
end


for K=1:10
for i=1:10
  centroids = initCentroids(X, K);
  indices = getClosestCentroids(X, centroids);
  centroids = computeCentroids(X, indices, K);
  iterations = 0;
        for ii = 1 :K
            clustering = X(find(indices == ii), :);
            cost = 0;
            for z = 1 : size(clustering,1)
                cost = cost + sum((clustering(z,:) - centroids(ii,:)).^2)/17999;
            end
            costVec(1,K) = cost;
            
        end
end
end       
[ o bestKvalue ] = min(costVec);
noClusters = 1:10;
plot(noClusters, costVec);


%%%%%


X1=mean(X);
X2=std(X);
for i=1:18
pdf(i)= normpdf(X(100,i),X1(i),X2(i));
end
output=prod(pdf);
