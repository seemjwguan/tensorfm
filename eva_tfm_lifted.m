function [res] = eva_tfm_lifted(X, w1, w2, U1, U2, V1, V2, k)

    res = w1' * X * w2;

    for s = 1:k

        res = res + 0.5 * (mat_innp(U1(:, s) * U2(:, s)', X) * mat_innp(V1(:, s) * V2(:, s)', X) - mat_innp((U1(:, s) * U2(:, s)') .* X, (V1(:, s) * V2(:, s)') .* X));

    end

end
