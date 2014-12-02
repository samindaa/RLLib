% Matlab - Eigen 3 test

data = load('housing.data');
[m, n] = size(data)

X = zeros(m, n-1); % first is biased
X(:,1:end) = data(:,1:end-1);
y = data(:,end);

% mean normalized
%X = bsxfun(@minus, X, mean(X));
%X = bsxfun(@rdivide, X, std(X, [], 1));
X = [ones(m,1), X];
% Regularized linear regression with normal equation

X(1,:)
y(1)

lambda = 1;

A_new = (X' * X + lambda * eye(size(X,2)));
b_new = X' * y;

theta_estimated = A_new \ b_new

mrse = sqrt(sum((y - X * theta_estimated).^2) / m);
sprintf('mrsq %f: ', mrse)

disp([y(1:10) X(1:10,:) * theta_estimated])