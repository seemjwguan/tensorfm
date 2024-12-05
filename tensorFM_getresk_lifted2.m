clear all;
addpath('./FMdata10x10sparse1000train')

dataset_name = 'fmnist';

Y = VChooseK(0:9, 2);
totalCell = cell(1, size(Y, 1));
% totalCell = cell(1, 9);

% genre1 = 3;
% genre2 = 7;
% for genre2 = 4:9
for data_ind = 1:size(Y, 1)

    genre1 = Y(data_ind, 1);
    genre2 = Y(data_ind, 2);
    % for data_ind = 0:9
    % data_name = strcat(dataset_name, string(data_ind))
    data_name = strcat(dataset_name, num2str(genre1), 'vs', num2str(genre2))
    para = load(strcat('./tensorFM_resk9_100iter_sparse1000train/lifted/', 'FM_', data_name, '_train', '.mat')).paraSubsets;

    valset = strcat('FM_', data_name, '_val');
    testset = strcat('FM_', data_name, '_test');
    Xval = load(strcat(valset, '_data.mat'));
    Xval = double(Xval.X);
    Xtest = load(strcat(testset, '_data.mat'));
    Xtest = double(Xtest.X);

    Tval = load(strcat(valset, '_label.mat'));
    Tval = double(Tval.T);
    Ttest = load(strcat(testset, '_label.mat'));
    Ttest = double(Ttest.T);

    [n1, n2, nval] = size(Xval);

    Xval_cell = cell(1, nval);
    yval_cell = cell(1, nval);

    for i = 1:nval
        Xval_cell{i} = Xval(:, :, i);
        yval_cell{i} = Tval(1, i);
    end

    [n1, n2, ntest] = size(Xtest); 

    Xtest_cell = cell(1, ntest);
    ytest_cell = cell(1, ntest);

    for i = 1:ntest
        Xtest_cell{i} = Xtest(:, :, i);
        ytest_cell{i} = Ttest(1, i);
    end

    paralist1 = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0];
    paralist2 = [5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0];
    klist = [8];
    dup = linspace(1, 5, 5);
    [para1, para2, para3, tau] = ndgrid(paralist1, paralist2, klist, dup);
    valresCell = cell(numel(paralist1), numel(paralist2), numel(klist));
    testresCell = cell(numel(paralist1), numel(paralist2), numel(klist));

    for ind1 = 1:numel(paralist1)

        for ind2 = 1:numel(paralist2)

            for ind3 = 1:numel(klist)

                temp_res = 0;

                for ind4 = 1:numel(dup)

                    beda_w = paralist1(ind1);
                    beda_P = paralist2(ind2);
                    k = klist(ind3);
                    ind = (ind4 - 1) * numel(paralist1) * numel(paralist2) * numel(klist) + (ind3 - 1) * numel(paralist1) * numel(paralist2) + (ind2 - 1) * numel(paralist1) + ind1;
                    % ind = (ind3 - 1) * numel(paralist) * numel(klist) + (ind2 - 1) * numel(paralist) + ind1;
                    % ind = (ind2 - 1) * numel(paralist) + ind1;
                    tempCell = para{ind, 1};
                    w1 = tempCell{1};
                    w2 = tempCell{2};
                    U1 = tempCell{3};
                    U2 = tempCell{4};
                    V1 = tempCell{5};
                    V2 = tempCell{6};
                    predicted = cell(1, nval);

                    % if beda_w == 1e-2 && beda_P == 1e-2
                    %     V1
                    %     pause(1999)
                    % end

                    parfor i = 1:nval
                        predicted{i} = eva_tfm_lifted(Xval_cell{i}, w1, w2, U1, U2, V1, V2, k);
                    end

                    temp_res = temp_res + FM_ACC(arrayfun(@(x) sign(x - 0), cell2mat(predicted)), arrayfun(@(x) sign(x - 0), cell2mat(yval_cell)));
                    % temp_res = temp_res + compute_RMSE(cell2mat(predicted), cell2mat(yval_cell));
                end

                valresCell{ind1, ind2, ind3} = temp_res / numel(dup);

            end

        end

    end

    for ind1 = 1:numel(paralist1)

        for ind2 = 1:numel(paralist2)

            for ind3 = 1:numel(klist)

                temp_res = 0;

                for ind4 = 1:numel(dup)

                    beda_w = paralist1(ind1);
                    beda_P = paralist2(ind2);
                    k = klist(ind3);
                    ind = (ind4 - 1) * numel(paralist1) * numel(paralist2) * numel(klist) + (ind3 - 1) * numel(paralist1) * numel(paralist2) + (ind2 - 1) * numel(paralist1) + ind1;
                    % ind = (ind3 - 1) * numel(paralist) * numel(klist) + (ind2 - 1) * numel(paralist) + ind1;
                    % ind = (ind2 - 1) * numel(paralist) + ind1;
                    tempCell = para{ind, 1};
                    w1 = tempCell{1};
                    w2 = tempCell{2};
                    U1 = tempCell{3};
                    U2 = tempCell{4};
                    V1 = tempCell{5};
                    V2 = tempCell{6};
                    predicted = cell(1, ntest);

                    parfor i = 1:ntest
                        predicted{i} = eva_tfm_lifted(Xtest_cell{i}, w1, w2, U1, U2, V1, V2, k);
                    end

                    temp_res = temp_res + FM_ACC(arrayfun(@(x) sign(x - 0), cell2mat(predicted)), arrayfun(@(x) sign(x - 0), cell2mat(ytest_cell)));
                    % temp_res = temp_res + compute_RMSE(cell2mat(predicted), cell2mat(ytest_cell));
                end

                testresCell{ind1, ind2, ind3} = temp_res / numel(dup);

            end

        end

    end

    valresCell = cell2mat(valresCell);
    valresCell = valresCell(:);
    testresCell = cell2mat(testresCell);
    testresCell = testresCell(:);

    maxval = max(valresCell);
    idx = find(valresCell==maxval);
    totalCell{data_ind} = max(testresCell(idx));
end

totalCell
save(strcat('./tensorFM_finalres/', dataset_name, '_lifted.mat'), 'totalCell')