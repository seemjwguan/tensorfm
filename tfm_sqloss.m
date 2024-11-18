classdef tfm_sqloss

    properties (Constant)

        mu = 1;

    end

    methods (Static = true)
        %% p: hat yi, y: real label.

        function l = loss(p, y)

            l = 0.5 * (p - y)^2;

        end

        function d = dloss(p, y)

            d = p - y;
            
        end

    end

end
