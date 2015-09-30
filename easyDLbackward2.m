function grad = easyDLbackward2(layers, a, a2, target, o)

    M = size(target, 2);
    
    grad = cell(numel(layers), 1);
    delta_adv1 = cell(numel(layers), 1);
    delta_adv2 = cell(numel(layers), 1);
    delta_adv1{end} = ( a{end-1} - a2{end-1} ) / M;
    delta_adv2{end} = -delta_adv1{end};

    global backward_conv_weight;
    global backward_conv_delta;

    for i = numel(layers):-1:1
        switch layers{i}.type
        case 'fc'

            if i < numel(layers)
                
                delta_adv1{i+1} = delta_adv1{i+1} .* layers{i}.derfun( a{i+1} );
                delta_adv2{i+1} = delta_adv2{i+1} .* layers{i}.derfun( a2{i+1} );
                
                % calculate gradient
                %tmp = norm(gradW(:));
                %fprintf('%f %f',tmp,o.weightdecay * norm(layers{i}.W(:)));
                grad{i}.W = o.manifold * ( delta_adv1{i+1} * a{i}' + delta_adv2{i+1} * a2{i}' );
                %fprintf(' %f => %f\n',norm(gradW(:))-tmp,norm(gradW(:)));
                grad{i}.b = o.manifold * ( sum(delta_adv1{i+1}, 2) + sum(delta_adv2{i+1}, 2) );
                
                % update delta
                delta_adv1{i} = layers{i}.W' * delta_adv1{i+1};
                delta_adv2{i} = layers{i}.W' * delta_adv2{i+1};
                delta_adv1(i+1) = [];
                delta_adv2(i+1) = [];
                % if the previous layer is convolutional, the delta needs to be reshaped
                if i > 1
                    if strcmp(layers{i-1}.type, 'pool') || strcmp(layers{i-1}.type, 'conv')
                        delta_adv1{i} = reshape(delta_adv1{i}, [layers{i-1}.outDim, M]);
                        delta_adv2{i} = reshape(delta_adv2{i}, [layers{i-1}.outDim, M]);
                    end
                end
            else
                grad{i}.W = zeros(size(layers{i}.W));
                grad{i}.b = zeros(size(layers{i}.b));
            end
            
        case 'pool'

            delta_adv1{i} = zeros([layers{i}.inDim, M]);
            delta_adv2{i} = zeros([layers{i}.inDim, M]);
            for r = 1:layers{i}.poolDim(1)
                for c = 1:layers{i}.poolDim(2)
                    delta_adv1{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta_adv1{i+1} / prod(layers{i}.poolDim);
                    delta_adv2{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta_adv2{i+1} / prod(layers{i}.poolDim);
                end
            end
            delta_adv1(i+1) = [];
            delta_adv2(i+1) = [];

        case 'conv'

            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, size(delta_adv1{i+1}));
            end
            delta_adv1{i+1} = delta_adv1{i+1} .* layers{i}.derfun( a{i+1} );
            delta_adv2{i+1} = delta_adv2{i+1} .* layers{i}.derfun( a2{i+1} );
            
            if i < numel(layers)
                gradW1 = backward_conv_weight( a{i}, delta_adv1{i+1}, layers{i}.ConvIdx, size(layers{i}.W) );
                gradW2 = backward_conv_weight( a2{i}, delta_adv2{i+1}, layers{i}.ConvIdx, size(layers{i}.W) );
                grad{i}.W = o.manifold * ( gradW1 + gradW2 );
                grad{i}.b = o.manifold * ( reshape(sum(sum(sum(delta_adv1{i+1},1),2),4),[],1) + ...
                    reshape(sum(sum(sum(delta_adv2{i+1},1),2),4),[],1) );
            else
                grad{i}.W = zeros(size(layers{i}.W));
                grad{i}.b = zeros(size(layers{i}.b));
            end
            
            if i > 1
                tdelta = cat(1, cat(1, layers{i}.vpadding, delta_adv1{i+1}), layers{i}.vpadding);
                tdelta = cat(2, cat(2, layers{i}.hpadding, tdelta), layers{i}.hpadding);
                delta_adv1{i} = backward_conv_delta( tdelta, layers{i}.W, layers{i}.BackConvIdx, [layers{i}.inDim, M] );
                tdelta = cat(1, cat(1, layers{i}.vpadding, delta_adv2{i+1}), layers{i}.vpadding);
                tdelta = cat(2, cat(2, layers{i}.hpadding, tdelta), layers{i}.hpadding);
                delta_adv2{i} = backward_conv_delta( tdelta, layers{i}.W, layers{i}.BackConvIdx, [layers{i}.inDim, M] );
                clear('tdelta');
            end
            delta_adv1(i+1) = [];
            delta_adv2(i+1) = [];

        end

    end