function [obj] = obj_tfm_ori2(X, y, w1, w2, U1, U2, k, n, beda_w, beda_P)

    obj = 0;

    for i = 1:n
        obj = obj + (1/n) * tfm_sqloss.loss(eva_tfm_ori(X{i}, w1, w2, U1, U2, k), y{i});
    end

    obj = obj + 0.5 * (beda_w * (norm(w1)^2 + norm(w2)^2) + beda_P * (norm(U1, 'fro')^2 + norm(U2, 'fro')^2));

end
