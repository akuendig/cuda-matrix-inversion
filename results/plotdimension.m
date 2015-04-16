function plotdimension(data, dim)

    data = data(data(:, 2) == dim, :);

    lu_blas_cpu = data(data(:, 4) == 1, :);
    lu_blas_omp_cpu = data(data(:, 4) == 2, :);
    chol_gpu = data(data(:, 4) == 3, :);
    gauss_batched_gpu = data(data(:, 4) == 4, :);
    lu_cuda_batched_gpu = data(data(:, 4) == 5, :);
    
    sortedplot(lu_blas_cpu, 1, 5, 3, 6, ':');
    hold on;
    sortedplot(lu_blas_omp_cpu, 1, 5, 3, 6, '-.');
    hold on;
    sortedplot(chol_gpu, 1, 5, 3, 6, '--');
    hold on;
    sortedplot(gauss_batched_gpu, 1, 5, 3, 6, '-');
    hold on;
    sortedplot(lu_cuda_batched_gpu, 1, 5, 3, 6, '-+');

    grid on;
    legend(...
        'LU inversion using LAPACK', ...
        'LU inversion using LAPACK and OpenMP with 8 threads', ...
        'LU inversion on GPU', ...
        'Gauss-Jordan inversion on GPU', ...
        'LU inversion on GPU using cuBLAS', ...
        'Location', 'NW');
    
    title(sprintf('Runtime with increasing number of %dx%d matrices', dim, dim));
    xlabel('Number of matrices');
    ylabel('Runtime in milliseconds (ms)');
        
    filename = sprintf('runtime_%dx%d.pdf', dim, dim);
    export_fig(filename);
    
    hold off;
    
end

