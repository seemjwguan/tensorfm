function [w1, w2, U1, U2, V1, V2] = tensorFM_lifted_syn2(X, y, n1, n2, k, beda_w, beda_P, verbose)
    w1 = 1 .* randn(n1, 1);
    w2 = 1 .* randn(n2, 1);
    U1 = 1 .* randn(n1, k);
    U2 = 1 .* randn(n2, k);
    V1 = 1 .* randn(n1, k);
    V2 = 1 .* randn(n2, k);
    n = length(X); % number of samples
    y_pred = zeros(n, 1);

    for i = 1:n
        y_pred(i, 1) = eva_tfm_lifted(X{i}, w1, w2, U1, U2, V1, V2, k);
    end

    % lrate = realmax; % learning rate-free
    iter = 0;
    maxIter = 500;
    last_obj = realmax;
    obj = obj_tfm_lifted2(X, y, w1, w2, U1, U2, V1, V2, k, n, beda_w, beda_P);

    while (iter < maxIter)
        viol = 0;

        %% Update w1
        for j = 1:n1
            grad_w1j = beda_w .* w1(j, 1);
            eta_w1j = beda_w;
            deriv_w1j = zeros(n, 1);

            for i = 1:n
                deriv_w1j(i, 1) = X{i}(j, :) * w2;
                grad_w1j = grad_w1j + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) .* deriv_w1j(i, 1);
                eta_w1j = eta_w1j + (1/n) * tfm_sqloss.mu * deriv_w1j(i, 1)^2;
            end

            update = (1 / eta_w1j) * grad_w1j;
            w1(j, 1) = w1(j, 1) - update;
            viol = viol + abs(update);

            for i = 1:n
                y_pred(i, 1) = y_pred(i, 1) - update * deriv_w1j(i, 1);
            end

        end

        %% Update w2
        for j = 1:n2
            grad_w2j = beda_w .* w2(j, 1);
            eta_w2j = beda_w;
            deriv_w2j = zeros(n, 1);

            for i = 1:n
                deriv_w2j(i, 1) = dot(X{i}(:, j), w1);
                grad_w2j = grad_w2j + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) .* deriv_w2j(i, 1);
                eta_w2j = eta_w2j + (1/n) * tfm_sqloss.mu * deriv_w2j(i, 1)^2;
            end

            update = (1 / eta_w2j) .* grad_w2j;
            w2(j, 1) = w2(j, 1) - update;
            viol = viol + abs(update);

            for i = 1:n
                y_pred(i, 1) = y_pred(i, 1) - update * deriv_w2j(i, 1);
            end

        end

        %% Precompute V1V2X
        innp_V1V2X = zeros(k, n);

        for s = 1:k

            for i = 1:n
                innp_V1V2X(s, i) = V1(:, s)' * X{i} * V2(:, s);
            end

        end

        %% Update U1
        for s = 1:k

            for j = 1:n1
                grad_u1js = beda_P .* U1(j, s);
                eta_u1js = beda_P;
                deriv_u1js = zeros(n, 1);

                for i = 1:n
                    deriv_u1js(i, 1) = 0.5 * (innp_V1V2X(s, i) * dot(X{i}(j, :)', U2(:, s)) - V1(j, s) * dot(U2(:, s), V2(:, s) .* X{i}(j, :)' .* X{i}(j, :)'));
                    grad_u1js = grad_u1js + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) * deriv_u1js(i, 1);
                    eta_u1js = eta_u1js + (1/n) * tfm_sqloss.mu * deriv_u1js(i, 1)^2;
                end

                update = (1 / eta_u1js) * grad_u1js;
                U1(j, s) = U1(j, s) - update;
                viol = viol + abs(update);

                for i = 1:n
                    y_pred(i, 1) = y_pred(i, 1) - update * deriv_u1js(i, 1);
                end

            end

        end

        %% Update U2
        for s = 1:k

            for j = 1:n2
                grad_u2js = beda_P .* U2(j, s);
                eta_u2js = beda_P;
                deriv_u2js = zeros(n, 1);

                for i = 1:n
                    deriv_u2js(i, 1) = 0.5 * (innp_V1V2X(s, i) * dot(X{i}(:, j), U1(:, s)) - V2(j, s) * dot(U1(:, s), V1(:, s) .* X{i}(:, j) .* X{i}(:, j)));
                    grad_u2js = grad_u2js + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) * deriv_u2js(i, 1);
                    eta_u2js = eta_u2js + (1/n) * tfm_sqloss.mu * deriv_u2js(i, 1)^2;
                end

                update = (1 / eta_u2js) * grad_u2js;
                U2(j, s) = U2(j, s) - update;
                viol = viol + abs(update);

                for i = 1:n
                    y_pred(i, 1) = y_pred(i, 1) - update * deriv_u2js(i, 1);
                end

            end

        end

        %% Precompute U1U2X
        innp_U1U2X = zeros(k, n);

        for s = 1:k

            for i = 1:n
                innp_U1U2X(s, i) = U1(:, s)' * X{i} * U2(:, s);
            end

        end

        %% Update V1
        for s = 1:k

            for j = 1:n1
                grad_v1js = beda_P .* V1(j, s);
                eta_v1js = beda_P;
                deriv_v1js = zeros(n, 1);

                for i = 1:n
                    deriv_v1js(i, 1) = 0.5 * (innp_U1U2X(s, i) * dot(X{i}(j, :)', V2(:, s)) - U1(j, s) * dot(V2(:, s), U2(:, s) .* X{i}(j, :)' .* X{i}(j, :)'));
                    grad_v1js = grad_v1js + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) * deriv_v1js(i, 1);
                    eta_v1js = eta_v1js + (1/n) * tfm_sqloss.mu * deriv_v1js(i, 1)^2;
                end

                update = (1 / eta_v1js) * grad_v1js;
                V1(j, s) = V1(j, s) - update;
                viol = viol + abs(update);

                for i = 1:n
                    y_pred(i, 1) = y_pred(i, 1) - update * deriv_v1js(i, 1);
                end

            end

        end

        %% Update V2
        for s = 1:k

            for j = 1:n2
                grad_v2js = beda_P .* V2(j, s);
                eta_v2js = beda_P;
                deriv_v2js = zeros(n, 1);

                for i = 1:n
                    deriv_v2js(i, 1) = 0.5 * (innp_U1U2X(s, i) * dot(X{i}(:, j), V1(:, s)) - U2(j, s) * dot(V1(:, s), U1(:, s) .* X{i}(:, j) .* X{i}(:, j)));
                    grad_v2js = grad_v2js + (1/n) * tfm_sqloss.dloss(y_pred(i, 1), y{i}) * deriv_v2js(i, 1);
                    eta_v2js = eta_v2js + (1/n) * tfm_sqloss.mu * deriv_v2js(i, 1)^2;
                end

                update = (1 / eta_v2js) * grad_v2js;
                V2(j, s) = V2(j, s) - update;
                viol = viol + abs(update);

                for i = 1:n
                    y_pred(i, 1) = y_pred(i, 1) - update * deriv_v2js(i, 1);
                end

            end

        end

        obj = obj_tfm_lifted2(X, y, w1, w2, U1, U2, V1, V2, k, n, beda_w, beda_P);

        if verbose
            fprintf("This is %d-th iteration, Obj is %f, viol is %f.\n", iter + 1, obj, viol);
        end

        if viol <= 1e-3
            break
        end

        iter = iter + 1;
        last_obj = obj;
    end

end
