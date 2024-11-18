function [res] = kernel_A2(p, x)

    res = 0.5 * (kernel_H2(p, x) - kernel_D2(p, x));

end
