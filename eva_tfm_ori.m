function [res] = eva_tfm_ori(X, w1, w2, U1, U2, k)

    res = w1' * X * w2;

    for s = 1:k
        res = res + kernel_A2(reshape(U1(:, s) * U2(:, s)', [], 1), reshape(X, [], 1));
    end

end
