function generate_matrices( path, dim, num, precision )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    dlmwrite(strcat(path, '/a.mats'), [num dim 1], 'delimiter', '\t');
    dlmwrite(strcat(path, '/b.mats'), [num dim dim], 'delimiter', '\t');
    dlmwrite(strcat(path, '/c.mats'), [num dim 1], 'delimiter', '\t');
    dlmwrite(strcat(path, '/d.mats'), [num dim 1], 'delimiter', '\t');
    dlmwrite(strcat(path, '/e.mats'), [num 1 1], 'delimiter', '\t');
    
    dlmwrite(strcat(path, '/means.mats'), [num 1 1], 'delimiter', '\t');
    dlmwrite(strcat(path, '/variances.mats'), [num 1 1], 'delimiter', '\t');

    for i=1:num
        A = rand(dim, 1);
        B = rand(dim, dim); % generate a random dim x dim matrix
        C = rand(dim, 1);
        D = rand(dim, 1);
        E = rand(1, 1);

        % construct a symmetric matrix using either
        B = B+B'; % OR B = B*B';
        % The first is significantly faster: O(dim^2) compared to O(dim^3)

        % since B(i,j) < 1 possible by construction and a symmetric
        % diagonally dominant matrix is symmetric positive definite, which
        % can be ensured by adding dim*I
        B = B + dim*eye(dim);

        dlmwrite(strcat(path, '/a.mats'), A, 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/b.mats'), B, 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/c.mats'), C, 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/d.mats'), D, 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/e.mats'), E, 'delimiter', '\t', '-append', 'precision', precision);

        dlmwrite(strcat(path, '/means.mats'), A'*(inv(B+diag(C))*D), 'delimiter', '\t', '-append', 'precision', precision);
        dlmwrite(strcat(path, '/variances.mats'), E-A'*(inv(B+diag(C))*A), 'delimiter', '\t', '-append', 'precision', precision);
    end
end

