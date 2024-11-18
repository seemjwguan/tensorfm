function [res] = mat_innp(A, B)
    res = sum(sum(A .* B));
end
