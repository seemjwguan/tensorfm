clear all;

dataset_name = "fmnist";
Y = VChooseK(0:9, 2);
for ind = 1:size(Y, 1)

    genre1 = Y(ind, 1);
    genre2 = Y(ind, 2);
    if isfile(strcat('./tensorFM_resk9_100iter_sparse1000train/orifm/', "FM_", dataset_name, string(genre1), "vs", string(genre2), "_train", '.mat'))
        continue
    end
    tensorFM_handler_subtasks_ori(dataset_name, genre1, genre2);

end
