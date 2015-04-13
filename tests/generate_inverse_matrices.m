function generate_inverse_matrices( path, dim, num, precision )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    mkdir(path);
    dlmwrite(strcat(path, '/a.mats'), [num dim dim], 'delimiter', '\t');
    dlmwrite(strcat(path, '/aInv.mats'), [num dim dim], 'delimiter', '\t');

    for i=1:num
        A = rand(dim, dim);

        % construct a symmetric matrix using either
        A = A+A'; % OR A = A*A';
        % The first is significantly faster: O(dim^2) compared to O(dim^3)

        % since A(i,j) < 1 possible by construction and a symmetric
        % diagonally dominant matrix is symmetric positive definite, which
        % can be ensured by adding dim*I
        A = A + dim*eye(dim);

        dlmwrite(strcat(path, '/a.mats'), A, 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/aInv.mats'), inv(A), 'delimiter', '\t', '-append', 'precision', precision);
    end
end

