function [grad, gradX] = easyDLbackward(layers, a, target, adversarial, o)

    M = size(target, 2);
    
    grad = cell(numel(layers), 1);
    delta = cell(numel(layers)+1, 1);
    delta{end} = (a{end} - target) / M;% .* a{end} .* (1-a{end});

    if adversarial
        delta_adv = cell(numel(layers)+1, 1);
        delta_adv{end} = (a{end} - target(:,randperm(M))) / M;% .* a{end} .* (1-a{end});
    end

    for i = numel(layers):-1:1
        switch layers{i}.type
        case 'fc'
if i == 0
delta{i+1}(a{i+1}<=0.5) = delta{i+1}(a{i+1}<=0.5) * 5;
end
            % calculate gradient
            gradW = delta{i+1} * a{i}' + o.weightdecay * 2 * layers{i}.W;
            grad{i}.W = gradW;
            gradb = sum(delta{i+1}, 2);
            grad{i}.b = gradb;
            
            if adversarial && i == 1
                gradX = layers{i}.W' * ( delta{i+1} .* a{i+1} .* a{i+1} );
            end

            % update delta
            delta{i} = layers{i}.W' * delta{i+1};
            if i < numel(layers)
                delta{i} = delta{i} .* a{i} .* (1-a{i});
            end
            delta(i+1) = [];

            % if the previous layer is convolutional, the delta needs to be reshaped
            if i > 1
                if strcmp(layers{i-1}.type, 'pool') || strcmp(layers{i-1}.type, 'conv')
                    delta{i} = reshape(delta{i}, [layers{i-1}.outDim, M]);
                end
            end

            if adversarial
                delta_adv{i} = layers{i}.W' * delta_adv{i+1};
                if i < numel(layers)
                    delta_adv{i} = delta_adv{i} .* a{i} .* (1-a{i});
                end
                delta_adv(i+1) = [];

                if i > 1
                    if strcmp(layers{i-1}.type, 'pool') || strcmp(layers{i-1}.type, 'conv')
                        delta_adv{i} = reshape(delta_adv{i}, [layers{i-1}.outDim, M]);
                    end
                end
            end

        case 'pool'

            delta{i} = zeros([layers{i}.inDim, M]);
            for r = 1:layers{i}.poolDim(1)
                for c = 1:layers{i}.poolDim(2)
                    delta{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta{i+1} / prod(layers{i}.poolDim);
                end
            end
            delta{i} = delta{i} .* a{i} .* (1-a{i});
            delta(i+1) = [];

            if adversarial
                delta_adv{i} = zeros([layers{i}.inDim, M]);
                for r = 1:layers{i}.poolDim(1)
                    for c = 1:layers{i}.poolDim(2)
                        delta_adv{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta_adv{i+1} / prod(layers{i}.poolDim);
                    end
                end
                delta_adv(i+1) = [];
            end

        case 'conv'

            J = layers{i}.inDim(3);
            K = layers{i}.outDim(3);
            
            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, size(delta{i+1}));
            end
            ndelta = flip(rot90(delta{i+1}, 2), 4);

            gradW = zeros(size(layers{i}.W));
            gradW_1norm = zeros(size(layers{i}.W));
            for j = 1:J
                for k = 1:K
                    if layers{i}.Conn(j,k)
                        gradW(:,:,j,k) = gradW(:,:,j,k) + convn(a{i}(:,:,j,:), ndelta(:,:,k,:), 'valid');
                        %for m = 1:M
                        %    gradW(:,:,j,k) = gradW(:,:,j,k) + conv2(a{i}(:,:,j,m), ndelta(:,:,k,m), 'valid');
                            %grad1W_1norm(:,:,j,k) = grad1W_1norm(:,:,j,k) + conv2(a{i}(:,:,j,:), tmpdelta(:,:,k,m), 'valid');
                        %end
                    end
                end
            end

            if adversarial && i == 1
                ndelta_adv = delta_adv{i+1} .* a{i+1} .* (1-a{i+1});
                ndelta_adv = rot90(ndelta_adv, 2);

                gradX = zeros(size(a{i}));
                for j = 1:J
                    for k = 1:K
                        if layers{i}.Conn(j,k)
                            tmpW = rot90(layers{i}.W(:,:,j,k),2);
                            for m = 1:M
                                gradX(:,:,j,m) = gradX(:,:,j,m) + conv2(ndelta_adv(:,:,k,m), tmpW);
                                %grad1W_1norm(:,:,j,k) = grad1W_1norm(:,:,j,k) + conv2(a{i}(:,:,j,:), tmpdelta(:,:,k,m), 'valid');
                            end
                        end
                    end
                end
            end

            gradW = gradW ...
                + o.weightdecay * layers{i}.W ...
                + o.sparsecoeff * gradW_1norm;
            gradb = reshape(sum(sum(sum(ndelta,1),2),4),[],1);
            grad{i}.W = gradW;
            grad{i}.b = gradb;

            if i > 1
                delta{i} = zeros([layers{i}.inDim, M]);
                for j = 1:J
                    for k = 1:K
                        if layers{i}.Conn(j,k)
                            delta{i}(:,:,j,:) = delta{i}(:,:,j,:) + convn(delta{i+1}(:,:,k,:), layers{i}.W(:,:,j,k));
                        end
                    end
                end
            end

        end

    end