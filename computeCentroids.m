function centroids = computeCentroids(X, idx, K)
[m n] = size(X);
centroids = zeros(K, n);
for i = 1:K
    c_i = idx==i;
    n_i = sum(c_i);
    c_i_matrix = repmat(c_i,1,n);
    X_c_i = X .* c_i_matrix;
    centroids(i,:) = sum(X_c_i) ./ n_i;
end
end

