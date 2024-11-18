function [] = tensorFM_handler_subtasks_ori(dataset_name, genre1, genre2)

    addpath('./FMdata10x10sparse1000train/');
    dataset = strcat("FM_", dataset_name, string(genre1), "vs", string(genre2), "_train");
    fprintf('Dataset is %s.\n', dataset);
    X = load(strcat(dataset, '_data.mat'));
    X = double(X.X);
    target = load(strcat(dataset, '_label.mat'));
    target = double(target.T);

    disp(size(X));

    %% specifying parameters.
    c = length(unique(target)); % number of classes
    fprintf('Number of classes is %d.\n', c);
    [n1, n2, n] = size(X);
    fprintf('Number of samples is %d.\n', n);

    X_cell = cell(1, n);
    y_cell = cell(1, n);

    for i = 1:n
        X_cell{i} = X(:, :, i);
        y_cell{i} = target(1, i);
    end

    % Determined. P may be useless.
    paralist1 = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0];
    paralist2 = [5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0];
    klist = [8];
    dup = linspace(1, 5, 5);
    [para1, para2, para3, tau] = ndgrid(paralist1, paralist2, klist, dup);
    paraSubsets = cell(numel(para1), 1);

    parfor ind = 1:numel(para1)
        rng(ind);
        beda_w = para1(ind);
        beda_P = para2(ind);
        k = para3(ind);
        fprintf('This is %d-th search. (total %d.) beta is %f and %f, k is %d.\n', ind, numel(para1), beda_w, beda_P, k);
        [w1, w2, U1, U2] = tensorFM_ori_ls2(X_cell, y_cell, n1, n2, k, beda_w, beda_P, false);
        tempCell = cell(1, 4);
        tempCell{1} = w1;
        tempCell{2} = w2;
        tempCell{3} = U1;
        tempCell{4} = U2;
        paraSubsets{ind, 1} = tempCell;
    end

    save(strcat('./tensorFM_resk9_100iter_sparse1000train/orifm/', dataset, '.mat'), 'paraSubsets');

end