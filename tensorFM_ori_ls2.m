function [w1, w2, U1, U2] = tensorFM_ori_ls2(X, y, n1, n2, k, beda_w, beda_P, verbose)
    w1 = 1 .* randn(n1, 1);
    w2 = 1 .* randn(n2, 1);
    U1 = 1 .* randn(n1, k);
    U2 = 1 .* randn(n2, k);
    n = length(X); % number of samples
    ls_initlrate = 1;
    ls_c = 1e-3;
    ls_gamma = 0.5;
    iter = 0;
    maxIter = 500;
    last_obj = realmax;
    obj = obj_tfm_ori2(X, y, w1, w2, U1, U2, k, n, beda_w, beda_P);
    % fprintf("Original obj is %f.\n", obj);

    while (iter < maxIter)
        viol = 0;

        %% Update w1
        for j = 1:n1
            grad_w1j = beda_w .* w1(j, 1);
            eta_w1j = beda_w;

            for i = 1:n
                deriv_w1j = X{i}(j, :) * w2;
                grad_w1j = grad_w1j + (1/n) * tfm_sqloss.dloss(eva_tfm_ori(X{i}, w1, w2, U1, U2, k), y{i}) .* deriv_w1j;
                eta_w1j = eta_w1j + (1/n) * tfm_sqloss.mu * deriv_w1j^2;
            end

            update = (1 / eta_w1j) * grad_w1j;
            w1(j, 1) = w1(j, 1) - update;
            viol = viol + abs(update);
        end

        %% Update w2
        for j = 1:n2
            grad_w2j = beda_w .* w2(j, 1);
            eta_w2j = beda_w;

            for i = 1:n
                deriv_w2j = dot(X{i}(:, j), w1);
                grad_w2j = grad_w2j + (1/n) * tfm_sqloss.dloss(eva_tfm_ori(X{i}, w1, w2, U1, U2, k), y{i}) .* deriv_w2j;
                eta_w2j = eta_w2j + (1/n) * tfm_sqloss.mu * deriv_w2j^2;
            end

            update = (1 / eta_w2j) .* grad_w2j;
            w2(j, 1) = w2(j, 1) - update;
            viol = viol + abs(update);
        end

        %% Update U1
        for s = 1:k
            grad_u1s = beda_P .* U1(:, s);

            for i = 1:n
                hat_yi = eva_tfm_ori(X{i}, w1, w2, U1, U2, k);
                allu_X = mat_innp(U1(:, s) * U2(:, s)', X{i});

                for j = 1:n1
                    grad_u1s(j, 1) = grad_u1s(j, 1) + (1/n) * tfm_sqloss.dloss(hat_yi, y{i}) .* (allu_X * dot(U2(:, s), X{i}(j, :)') - U1(j, s) * norm(U2(:, s) .* X{i}(j, :)')^2);
                end

            end

            %% line search
            temp_lrate = ls_initlrate;
            val_f0 = obj_tfm_ori2(X, y, w1, w2, U1, U2, k, n, beda_w, beda_P);

            while true

                U1new = U1;
                U1new(:, s) = U1new(:, s) - temp_lrate .* grad_u1s;
                val_f0_new = obj_tfm_ori2(X, y, w1, w2, U1new, U2, k, n, beda_w, beda_P);

                if (val_f0_new <= val_f0 - ls_c * temp_lrate * norm(grad_u1s)^2)
                    break
                else
                    temp_lrate = temp_lrate * ls_gamma;
                end

            end

            %% line search
            % temp_lrate

            update = temp_lrate .* grad_u1s;
            U1(:, s) = U1(:, s) - update;
            viol = viol + norm(update, 1);

        end

        %% Update U2
        for s = 1:k
            grad_u2s = beda_P .* U2(:, s);

            for i = 1:n
                hat_yi = eva_tfm_ori(X{i}, w1, w2, U1, U2, k);
                allu_X = mat_innp(U1(:, s) * U2(:, s)', X{i});

                for j = 1:n2
                    grad_u2s(j, 1) = grad_u2s(j, 1) + (1/n) * tfm_sqloss.dloss(hat_yi, y{i}) .* (allu_X * dot(U1(:, s), X{i}(:, j)) - U2(j, s) * norm(U1(:, s) .* X{i}(:, j))^2);
                end

            end

            %% line search
            temp_lrate = ls_initlrate;
            val_f0 = obj_tfm_ori2(X, y, w1, w2, U1, U2, k, n, beda_w, beda_P);

            while true

                U2new = U2;
                U2new(:, s) = U2new(:, s) - temp_lrate .* grad_u2s;
                val_f0_new = obj_tfm_ori2(X, y, w1, w2, U1, U2new, k, n, beda_w, beda_P);

                if (val_f0_new <= val_f0 - ls_c * temp_lrate * norm(grad_u2s)^2)
                    break
                else
                    temp_lrate = temp_lrate * ls_gamma;
                end

            end

            %% line search
            % temp_lrate

            update = temp_lrate .* grad_u2s;
            U2(:, s) = U2(:, s) - update;
            viol = viol + norm(update, 1);

        end

        obj = obj_tfm_ori2(X, y, w1, w2, U1, U2, k, n, beda_w, beda_P);
        % fprintf("This is %d-th iteration, obj is %f.\n", iter+1, obj);

        if verbose
            fprintf("This is %d-th iteration, Obj is %f, viol is %f.\n", iter + 1, obj, viol);
        end

        % if (last_obj - obj) / obj <= 1e-6
        %     break
        % end

        % fprintf("%f, %f.\n", w1' * X{1} * w2, kernel_A2(reshape(U1(:, 1) * U2(:, 1)', [], 1), reshape(X{1}, [], 1)));

        if viol <= 1e-3
            break
        end

        iter = iter + 1;
        last_obj = obj;
    end

end
