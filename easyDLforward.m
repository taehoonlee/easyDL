function a = easyDLforward(layers, testdata)

    global forward_conv;

    a = cell(numel(layers)+1, 1);
    a{1} = testdata;
    M = size(testdata); M = M(end);
    
    for i = 1:numel(layers)
        if isfield(layers{i}, 'mask') && ~isempty(layers{i}.mask)
            a{i} = a{i} .* layers{i}.mask;
        end
        switch layers{i}.type
        case 'conv'
            if isfield(layers{i}, 'ConvIdx') && prod([layers{i}.outDim(1:2), M]) == size(layers{i}.ConvIdx, 2)
                a{i+1} = forward_conv( a{i}, layers{i}.W, layers{i}.b, layers{i}.Conn, layers{i}.ConvIdx, [layers{i}.outDim(1:2), M] );
            else
                a{i+1} = cnnConvolve( a{i}, layers{i}.W, layers{i}.b, layers{i}.Conn );
            end
            a{i+1} = layers{i}.actfun( a{i+1} );
            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, prod(layers{i}.outDim), []);
            end
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
            aux1 = bsxfun(@plus, layers{i}.W * a{i}, layers{i}.b);
            if i == numel(layers) % if a current layer is the last, perform the softmax operation.
                aux3 = exp(bsxfun(@minus, aux1, max(aux1, [], 1)));
                a{i+1} = bsxfun(@rdivide, aux3, sum(aux3));
            else % otherwise, perform sigmoid operation.
                %aux1(aux1<0) = aux1(aux1<0) / 5;
                a{i+1} = layers{i}.actfun( aux1 );
            end
        case 'ae'
            a{i+1} = cell(2, 1);
            a{i+1}{1} = 1 ./ ( 1 + exp(-bsxfun(@plus, layers{i}.W1 * a{i}, layers{i}.b1)) );
            a{i+1}{2} = max(bsxfun(@plus, layers{i}.W2 * a{i+1}{1}, layers{i}.b2), 0);
        end
    end