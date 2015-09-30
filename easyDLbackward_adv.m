function gradX = easyDLbackward_adv(layers, a, target)

    M = size(target, 2);
    
    delta_adv = cell(numel(layers)+1, 1);
    delta_adv{end} = (a{end} - target(:,randperm(M))) / M;% .* a{end} .* (1-a{end});

    global backward_conv_delta;

    for i = numel(layers):-1:1
        switch layers{i}.type
        case 'fc'
            
            if i < numel(layers)
                delta_adv{i+1} = delta_adv{i+1} .* layers{i}.derfun( a{i+1} );
            end
            
            if i == 1
                gradX = layers{i}.W' * delta_adv{i+1};
            end

            delta_adv{i} = layers{i}.W' * delta_adv{i+1};
            delta_adv(i+1) = [];

            if i > 1
                if strcmp(layers{i-1}.type, 'pool') || strcmp(layers{i-1}.type, 'conv')
                    delta_adv{i} = reshape(delta_adv{i}, [layers{i-1}.outDim, M]);
                end
            end

        case 'pool'

            delta_adv{i} = zeros([layers{i}.inDim, M]);
            for r = 1:layers{i}.poolDim(1)
                for c = 1:layers{i}.poolDim(2)
                    delta_adv{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta_adv{i+1} / prod(layers{i}.poolDim);
                end
            end
            delta_adv(i+1) = [];

        case 'conv'

            J = layers{i}.inDim(3);
            K = layers{i}.outDim(3);
            
            if strcmp(layers{i+1}.type, 'fc')
                a{i+1} = reshape(a{i+1}, size(delta_adv{i+1}));
            end
            
            if i == 1
                ndelta_adv = delta_adv{i+1} .* layers{i}.derfun( a{i+1} );
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
            
            if i > 1
                tdelta = cat(1, cat(1, layers{i}.vpadding, delta_adv{i+1}), layers{i}.vpadding);
                tdelta = cat(2, cat(2, layers{i}.hpadding, tdelta), layers{i}.hpadding);
                delta_adv{i} = backward_conv_delta( tdelta, layers{i}.W, layers{i}.BackConvIdx, [layers{i}.inDim, M] );
                clear('tdelta');
            end
            delta_adv(i+1) = [];
            
        end

    end