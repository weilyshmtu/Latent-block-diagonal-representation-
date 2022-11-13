function Lx = computeL(X,neighborsize)
options = [];
options.NeighborMode = 'KNN';
options.k = neighborsize;
options.WeightMode = 'Binary';
W = constructW(X',options);
[M] = constructM(X,4);
W = W.*M;
D = diag(sum(W,2));
Lx = D - W;% Lx = eye(size(X,2)) - D^(-0.5)*W*D^(-0.5);
