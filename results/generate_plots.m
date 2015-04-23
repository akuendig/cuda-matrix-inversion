gaussbench = dataset('File', 'gauss-bench.txt', 'Delimiter', ',', 'VarNames', {'timer', 'numMatrices', 'numDimensions', 'timeMS', 'timeNS'}, 'ReadVarNames', false);

gauss_stat = grpstats(gaussbench, {'timer', 'numMatrices', 'numDimensions'}, {'mean', 'std'}, 'DataVars', {'timeNS'});
gauss_stat_timer = dataset2cell(gauss_stat(:, 'timer'));
gauss_stat_timer = gauss_stat_timer(2:end, 2);

lines = {
    ':'
    '-.'
    '--'
    '-'
    '-+'
    '-x'
};

mkdir('plots');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the different parts that make  up MEAN computation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot a stacked bar plot for the different amount of time used in each
% part of the gauss benchmark
mean_perf_vars = {
    'calculate_mean_gpu_mem_htod'
    'calculate_mean_gpu_add'
    'calculate_mean_gpu_inv'
    'calculate_mean_gpu_mul'
    'calculate_mean_gpu_dot'
    'calculate_mean_gpu_mem_dtoh'
};

mean_ds = gauss_stat(ismember(gauss_stat_timer, mean_perf_vars), :);

%for i=[1 2 4 8 16]
for dup=[16]
    ds = mean_ds(mean_ds.numMatrices == dup*100, :);
    ds = [
        double(ds(ds.numDimensions == 8, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 16, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 32, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 64, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 128, 'mean_timeNS'))'
    ];
    ds = ds / 1000000;

    figure;
    % bar([8 16 32 64 128], ds, 'stacked');
    bar(ds, 'stacked');
    grid on
    set(gca,'XTickLabel',{'8', '16', '32', '64', '128'});
    xlabel('matrix dimension');
    ylabel('Runtime in milliseconds (ms)');
    legend(...
        'Transfer H->D', 'Addition', 'Inversion',...
        'Multiplication', 'Dot Product', 'Transfer D->H',...
        'Location', 'NorthWest');
    export_fig(sprintf('plots/partial_mean_%d.pdf', dup));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the different parts that make  up VARIANCE computation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

variance_perf_vars = {
    'calculate_variance_gpu_mem_htod'
    'calculate_variance_gpu_add'
    'calculate_variance_gpu_inv'
    'calculate_variance_gpu_mul'
    'calculate_variance_gpu_dot'
    'calculate_variance_gpu_mem_dtoh'
};

variance_ds = gauss_stat(ismember(gauss_stat_timer, variance_perf_vars), :);

%for i=[1 2 4 8 16]
for dup=[16]
    ds = variance_ds(variance_ds.numMatrices == dup*100, :);
    ds = [
        double(ds(ds.numDimensions == 8, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 16, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 32, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 64, 'mean_timeNS'))'
        double(ds(ds.numDimensions == 128, 'mean_timeNS'))'
    ];
    ds = ds / 1000000;

    figure;
    % bar([8 16 32 64 128], ds, 'stacked');
    bar(ds, 'stacked');
    grid on
    set(gca,'XTickLabel',{'8', '16', '32', '64', '128'});
    xlabel('matrix dimension');
    ylabel('Runtime in milliseconds (ms)');
    legend(...
        'Transfer H->D', 'Addition', 'Inversion',...
        'Multiplication', 'Dot Product', 'Transfer D->H',...
        'Location', 'NorthWest');
    export_fig(sprintf('plots/partial_variance_%d.pdf', dup));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the total runtime of the CPU and GPU computation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for dim=[16 128]
    figure;
    i = 1;
    
    for j={'means_cpu', 'variances_cpu', 'means_gpu', 'variances_gpu'}
        ds = gauss_stat(strcmp(gauss_stat.timer, j) & gauss_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1), cell2mat(lines(i)));
        hold on
        i = i + 1;
    end

    grid on
    legend('Mean on CPU', 'Variance on CPU', 'Mean on GPU', 'Variance on GPU', 'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
    export_fig(sprintf('plots/gauss_parallel_%dx%d.pdf', dim, dim));
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the gaussian runtime for 1 thread %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gaussbench = dataset('File', 'gauss-bench-single.txt', 'Delimiter', ',', 'VarNames', {'timer', 'numMatrices', 'numDimensions', 'timeMS', 'timeNS'}, 'ReadVarNames', false);

gauss_stat = grpstats(gaussbench, {'timer', 'numMatrices', 'numDimensions'}, {'mean', 'std'}, 'DataVars', {'timeNS'});
gauss_stat_timer = dataset2cell(gauss_stat(:, 'timer'));
gauss_stat_timer = gauss_stat_timer(2:end, 2);

for dim=[16 128]
    figure;
    i = 1;
    
    for j={'means_cpu', 'variances_cpu', 'means_gpu', 'variances_gpu'}
        ds = gauss_stat(strcmp(gauss_stat.timer, j) & gauss_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1), cell2mat(lines(i)));
        hold on
        i = i + 1;
    end

    grid on
    legend('Mean on CPU', 'Variance on CPU', 'Mean on GPU', 'Variance on GPU', 'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
    export_fig(sprintf('plots/gauss_single_%dx%d.pdf', dim, dim));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the inversion runtime %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inversebench = dataset('File', 'inverse-bench.txt', 'Delimiter', ',', 'VarNames', {'timer', 'numMatrices', 'numDimensions', 'timeMS', 'timeNS'}, 'ReadVarNames', false);
inv_stat = grpstats(inversebench, {'timer', 'numMatrices', 'numDimensions'}, {'mean', 'std'}, 'DataVars', {'timeNS'});

for dim=[8 32 128]
    figure;
    i = 1;
    
    for j={'lu_blas_cpu', 'lu_blas_omp_cpu', 'chol_gpu', 'chol_mm2_gpu', 'gauss_batched_gpu', 'lu_cuda_batched_gpu'}
        ds = inv_stat(strcmp(inv_stat.timer, j) & inv_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1), cell2mat(lines(i)));
        hold on
        i = i + 1;
    end

    grid on
    legend(...
        'Cholesky Decomposition on CPU', 'Cholesky Decomposition on CPU 8 threads',...
        'Cholesky Decomposition on GPU', 'Cholesky Decomposition on GPU v2',...
        'Gauss-Jordan Inversion on GPU', 'LU factorization using cuBLAS',...
        'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
    export_fig(sprintf('plots/inversion_%dx%d.pdf', dim, dim));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the partial inversion runtime %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

algo_names = {
    'Cholesky Decomposition'
    'Cholesky Decomposition v2'
    'Gauss-Jordan Inversion'
    'LU factorization using cuBLAS'
};
    

%for i=[1 2 4 8 16]
for dup=[16]
    i = 1;
    
    for j={'decompose_cholesky_batched_gpu', 'cholesky_mm2_batched_gpu', 'inverse_gauss_batched_gpu', 'inverse_lu_cuda_batched_gpu', }
        ds = inv_stat(inv_stat.numMatrices == dup*100, :);
        htod = ds(strcmp(ds.timer, sprintf('%s_mem_htod', cell2mat(j))), :);
        ker = ds(strcmp(ds.timer, sprintf('%s_ker', cell2mat(j))), :);
        dtoh = ds(strcmp(ds.timer, sprintf('%s_mem_dtoh', cell2mat(j))), :);

        ds = [
            double(htod(htod.numDimensions == 8, 'mean_timeNS')) double(ker(ker.numDimensions == 8, 'mean_timeNS')) double(dtoh(dtoh.numDimensions == 8, 'mean_timeNS'))
            double(htod(htod.numDimensions == 16, 'mean_timeNS')) double(ker(ker.numDimensions == 16, 'mean_timeNS')) double(dtoh(dtoh.numDimensions == 16, 'mean_timeNS'))
            double(htod(htod.numDimensions == 32, 'mean_timeNS')) double(ker(ker.numDimensions == 32, 'mean_timeNS')) double(dtoh(dtoh.numDimensions == 32, 'mean_timeNS'))
            double(htod(htod.numDimensions == 64, 'mean_timeNS')) double(ker(ker.numDimensions == 64, 'mean_timeNS')) double(dtoh(dtoh.numDimensions == 64, 'mean_timeNS'))
            double(htod(htod.numDimensions == 128, 'mean_timeNS')) double(ker(ker.numDimensions == 128, 'mean_timeNS')) double(dtoh(dtoh.numDimensions == 128, 'mean_timeNS'))
        ];
        ds = ds / 1000000;

        figure;
        % bar([8 16 32 64 128], ds, 'stacked');
        bar(ds, 'stacked');
        grid on
        set(gca,'XTickLabel',{'8', '16', '32', '64', '128'});
        xlabel('Matrix dimension');
        ylabel('Runtime in milliseconds (ms)');
        legend(...
            'Transfer H->D', cell2mat(algo_names(i)), 'Transfer D->H',...
            'Location', 'NorthWest');
        export_fig(sprintf('plots/partial_inversion_%s.pdf', cell2mat(j)));
        i = i + 1;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot error of gaussian benchmark %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gausserror = dataset('File', 'gauss-bench-errors.txt', 'Delimiter', ' ', 'VarNames', {'numMatrices', 'numDimensions', 'numDup', 'timer', 'timeMS', 'err'}, 'ReadVarNames', false);

for dup=[16]
    ds = gausserror(gausserror.numMatrices == dup*100, :);
    ds = [
        %double(ds(ds.numDimensions == 8, 'err'))'
        %double(ds(ds.numDimensions == 16, 'err'))'
        %double(ds(ds.numDimensions == 32, 'err'))'
        %double(ds(ds.numDimensions == 64, 'err'))'
        double(ds(ds.numDimensions == 128, 'err'))
    ];

    figure;
    % bar([8 16 32 64 128], ds, 'stacked');
    bar(ds);
    grid on
    set(gca,'XTickLabel',{'Mean on CPU', 'Variance on CPU', 'Mean on GPU', 'Variance on GPU'});
    ylabel('Average error per matrix');
    export_fig(sprintf('plots/gauss-errors_%d.pdf', dup));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot error of inversion benchmark %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inverseerror = dataset('File', 'inverse-bench-errors.txt', 'Delimiter', ' ', 'VarNames', {'numMatrices', 'numDimensions', 'numDup', 'timer', 'timeMS', 'err'}, 'ReadVarNames', false);

for dup=[16]
    ds = inverseerror(inverseerror.numMatrices == dup*100, :);
    ds = ds(strcmp(ds.timer, 'chol_mm2_gpu') == 0, :);
    ds = ds(ds.numDimensions == 128, :);
    ds = [
        double(ds(strcmp(ds.timer, 'chol_gpu'), 'err')) double(ds(strcmp(ds.timer, 'lu_cuda_batched_gpu'), 'err'))
    ];

    sprintf('Mean error for cholesky decomposition: %f', ds(1))
    sprintf('Mean error for gauss-jordan: %f', ds(2))
end
