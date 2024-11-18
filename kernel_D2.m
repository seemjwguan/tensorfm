function [res] = kernel_D2(p, x)

    res = norm(p .* x)^2;

end
