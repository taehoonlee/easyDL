function a = easyDLforward(layers, testdata)
    a = cell(numel(layers)+1, 1);
    a{1} = testdata;
    for i = 1:numel(layers)
        switch layers{i}.type
        case 'conv'
            a{i+1} = cnnConvolve(a{i}, layers{i}.W, layers{i}.b, layers{i}.Conn);
        case 'pool'
%                 a{i+1} = reshape( ...
%                     mean(reshape( ...
%                     mean(reshape(a{i},layers{i}.poolDim(1),[]),1) ...
%                     ,layers{i}.outDim(1),layers{i}.poolDim(2),[]),2) ...
%                     , layers{i}.outDim(1), layers{i}.outDim(2), layers{i}.outDim(3), []);
            a{i+1} = zeros(size(a{i}) ./ [layers{i}.poolDim, 1, 1]);
            if strcmp(layers{i}.pooling, 'max')
                for r = 1:layers{i}.poolDim(1)
                    for c = 1:layers{i}.poolDim(2)
                        a{i+1} = max(a{i+1}, a{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:));
                    end
                end
            elseif strcmp(layers{i}.pooling, 'mean')
                for r = 1:layers{i}.poolDim(1)
                    for c = 1:layers{i}.poolDim(2)
                        a{i+1} = a{i+1} ...
                            + a{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:);
                    end
                end
                a{i+1} = a{i+1} / prod(layers{i}.poolDim);
            end
            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, prod(layers{i}.outDim), []);
            end
        case 'fc'
            if layers{i}.dropout > 0
                a{i} = a{i} .* (rand(size(a{i})) > layers{i}.dropout);
            end
            if i == 1
                a{i} = reshape(a{i}, prod(layers{i}.inDim), []);
            end
            aux1 = bsxfun(@plus, layers{i}.W * a{i}, layers{i}.b);
            if i == numel(layers) % if a current layer is the last, perform the softmax operation.
                aux3 = exp(bsxfun(@minus, aux1, max(aux1, [], 1)));
                a{i+1} = bsxfun(@rdivide, aux3, sum(aux3));
            else % otherwise, perform sigmoid operation.
                a{i+1} = 1 ./ ( 1 + exp(-aux1) );
            end
        end
    end