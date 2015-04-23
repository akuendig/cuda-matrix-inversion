gaussbench = dataset('File', 'gauss-bench.txt', 'Delimiter', ',', 'VarNames', {'timer', 'numMatrices', 'numDimensions', 'timeMS', 'timeNS'}, 'ReadVarNames', false);
inversebench = dataset('File', 'inverse-bench.txt', 'Delimiter', ',', 'VarNames', {'timer', 'numMatrices', 'numDimensions', 'timeMS', 'timeNS'}, 'ReadVarNames', false);

gauss_stat = grpstats(gaussbench, {'timer', 'numMatrices', 'numDimensions'}, {'mean', 'std'}, 'DataVars', {'timeNS'});
gauss_stat_timer = dataset2cell(gauss_stat(:, 'timer'));
gauss_stat_timer = gauss_stat_timer(2:end, 2);

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
for i=[16]
    ds = mean_ds(mean_ds.numMatrices == i*100, :);
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
        'Transfer H->D', 'Addidion', 'Inversion',...
        'Multiplication', 'Dot Product', 'Transfer D->H',...
        'Location', 'NorthWest');
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
for i=[16]
    ds = variance_ds(variance_ds.numMatrices == i*100, :);
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
    ylabel('Runtime in milliseconds (ms)');
    legend(...
        'Transfer H->D', 'Addidion', 'Inversion',...
        'Multiplication', 'Dot Product', 'Transfer D->H',...
        'Location', 'NorthWest');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the total runtime of the CPU and GPU computation %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for dim=[8 32 128]
    figure
    
    for j={'means_cpu', 'variances_cpu', 'means_gpu', 'variances_gpu'}
        ds = gauss_stat(strcmp(gauss_stat.timer, j) & gauss_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1));
        hold on
    end

    grid on
    legend('Mean on CPU', 'Variance on CPU', 'Mean on GPU', 'Mean on CPU', 'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the inversion runtime %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for dim=[8 32 128]
    figure
    
    for j={'means_cpu', 'variances_cpu', 'means_gpu', 'variances_gpu'}
        ds = gauss_stat(strcmp(gauss_stat.timer, j) & gauss_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1));
        hold on
    end

    grid on
    legend('Mean on CPU', 'Variance on CPU', 'Mean on GPU', 'Mean on CPU', 'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
end

inv_stat = grpstats(inversebench, {'timer', 'numMatrices', 'numDimensions'}, {'mean', 'std'}, 'DataVars', {'timeNS'});

for dim=[8 32 128]
    figure

    for j={'lu_blas_cpu', 'lu_blas_omp_cpu', 'chol_gpu', 'chol_mm2_gpu', 'gauss_batched_gpu', 'lu_cuda_batched_gpu'}
        ds = inv_stat(strcmp(inv_stat.timer, j) & inv_stat.numDimensions == dim, :);

        means = double(ds(:, 'mean_timeNS')) / 1000000;
        sem = double(ds(:, 'std_timeNS')) / 1000000 ./ sqrt(double(ds(:, 'GroupCount')));

        errorbar([1 2 4 8 16]*100, means, sem.*tinv(0.995, double(ds(:, 'GroupCount')) - 1));
        hold on
    end

    grid on
    legend(...
        'LU Inversion on CPU', 'LU Inverson on CPU 8 threads',...
        'Cholesky Inversion on GPU', 'Cholesky Inversion on GPU v2',...
        'Gauss-Jordan Inversion on GPU', 'Cholesky Inversion using cuBLAS',...
        'Location', 'NorthWest');
    xlabel(sprintf('Number of %dx%d matrices', dim, dim));
    ylabel('Runtime in milliseconds (ms)');
    hold off
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the partial inversion runtime %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%for i=[1 2 4 8 16]
for i=[16]
    for j={'cholesky_mm2_batched_gpu', 'decompose_cholesky_batched_gpu', 'inverse_gauss_batched_gpu', 'inverse_lu_cuda_batched_gpu', }
        ds = inv_stat(inv_stat.numMatrices == i*100, :);
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
            'Transfer H->D', sprintf('Inversion using %s', strrep(cell2mat(j), '_', '\_')), 'Transfer D->H',...
            'Location', 'NorthWest');
    end
end
