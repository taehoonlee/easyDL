function grad = easyDLbackward(layers, a, target, o)

    M = size(target, 2);
    
    grad = cell(numel(layers), 1);
    delta = cell(numel(layers)+1, 1);
    delta{end} = (a{end} - target) / M;% .* a{end} .* (1-a{end});
    
    global backward_conv_weight;
    global backward_conv_delta;

    for i = numel(layers):-1:1
        switch layers{i}.type
        case 'fc'
if i == 0
delta{i+1}(a{i+1}<=0.5) = delta{i+1}(a{i+1}<=0.5) * 5;
end
            if i < numel(layers)
                delta{i+1} = delta{i+1} .* layers{i}.derfun( a{i+1} );
            end
            
            % calculate gradient
            grad{i}.W = delta{i+1} * a{i}' + o.weightdecay * layers{i}.W;
            grad{i}.b = sum(delta{i+1}, 2);
            
            % update delta
            delta{i} = layers{i}.W' * delta{i+1};
            if layers{i}.dropout > 0
                delta{i} = delta{i} .* layers{i}.mask;
            end
            delta(i+1) = [];

            % if the previous layer is convolutional, the delta needs to be reshaped
            if i > 1
                if strcmp(layers{i-1}.type, 'pool') || strcmp(layers{i-1}.type, 'conv')
                    delta{i} = reshape(delta{i}, [layers{i-1}.outDim, M]);
                end
            end

        case 'pool'

            delta{i} = zeros([layers{i}.inDim, M]);
            for r = 1:layers{i}.poolDim(1)
                for c = 1:layers{i}.poolDim(2)
                    delta{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta{i+1} / prod(layers{i}.poolDim);
                end
            end
            delta(i+1) = [];

        case 'conv'

            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, size(delta{i+1}));
            end
            delta{i+1} = delta{i+1} .* layers{i}.derfun( a{i+1} );
            
            gradW_1norm = zeros(size(layers{i}.W));
            gradW = backward_conv_weight( a{i}, delta{i+1}, layers{i}.ConvIdx, size(layers{i}.W) );
            %gradW = backward_conv_weight_old( a{i}, delta{i+1}, layers{i}.Conn, size(layers{i}.W) );
            
            grad{i}.W = gradW + o.weightdecay * layers{i}.W + o.sparsecoeff * gradW_1norm;
            grad{i}.b = reshape(sum(sum(sum(delta{i+1},1),2),4),[],1);

            if i > 1
                tdelta = cat(1, cat(1, layers{i}.vpadding, delta{i+1}), layers{i}.vpadding);
                tdelta = cat(2, cat(2, layers{i}.hpadding, tdelta), layers{i}.hpadding);
                delta{i} = backward_conv_delta( tdelta, layers{i}.W, layers{i}.BackConvIdx, [layers{i}.inDim, M] );
                %delta{i} = backward_conv_delta_old( delta{i+1}, layers{i}.W, layers{i}.Conn, [layers{i}.inDim, M] );
                clear('tdelta');
            end
            delta(i+1) = [];

        end

    end
    
end



function out = backward_conv_weight_old( a, delta, conn, outsize )
    
    ndelta = flip(rot90(delta, 2), 4);
    out = zeros(outsize);
    J = outsize(3);
    K = outsize(4);
    for k = 1:K
        if sum(conn) == J
            out(:,:,:,k) = convn(a, ndelta(:,:,k,:), 'valid');
        else
            for j = 1:J
                if conn(j,k)
                    out(:,:,j,k) = convn(a(:,:,j,:), ndelta(:,:,k,:), 'valid');
                    %for m = 1:M
                    %    gradW(:,:,j,k) = gradW(:,:,j,k) + conv2(a{i}(:,:,j,m), ndelta(:,:,k,m), 'valid');
                        %grad1W_1norm(:,:,j,k) = grad1W_1norm(:,:,j,k) + conv2(a{i}(:,:,j,:), tmpdelta(:,:,k,m), 'valid');
                    %end
                end
            end
        end
    end
    
end

function delta_bottom = backward_conv_delta_old( delta_top, weight, conn, outsize )
    
    delta_bottom = zeros(outsize);
    J = outsize(3);
    K = size(delta_top, 3);
    for j = 1:J
        for k = 1:K
            if conn(j,k)
                delta_bottom(:,:,j,:) = delta_bottom(:,:,j,:) + convn(delta_top(:,:,k,:), weight(:,:,j,k));
            end
        end
    end
    
end