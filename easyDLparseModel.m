function layers = easyDLparseModel(theta, numRows, numCols, numChannels, varargin)

    if nargin > 4
        numClasses = varargin{1};
    end
    
    layers = cell(numel(theta), 1);
    
    for c = 1:numel(theta)
        tmp = regexp(theta{c}, ',', 'split');
        info = regexp(tmp{1}, '\d+', 'match');
        switch lower(theta{c}(1))
            case 'c'

                layers{c}.type = 'conv';
                numOutFilters = str2double(info(1));
                filterDim = str2double(info(2:3));
                if c == 1
                    layers{c}.inDim = [numRows, numCols, numChannels];
                else
                    layers{c}.inDim = layers{c-1}.outDim;
                end
                layers{c}.outDim = [layers{c}.inDim(1:2) - filterDim + 1, numOutFilters];

                layers{c}.W = 0.1 * randn([filterDim, layers{c}.inDim(3), layers{c}.outDim(3)]);
                layers{c}.b = zeros(numOutFilters, 1);
                layers{c}.numFilters = numOutFilters;

                if numel(tmp) > 1
                    if strcmpi(tmp{2}, 'sparseconn')
                        inMap = layers{c}.inDim(3);
                        tmp1 = logical(eye(inMap));
                        tmp2 = [tmp1; zeros(3, inMap)] | ...
                            [zeros(3, inMap); tmp1];
                        tmp2(1:3,:) = tmp2(1:3,:) | tmp2(end-2:end,:);
                        tmp2(end-2:end,:) = [];
                        tmp3 = [tmp1; zeros(2, inMap)] | ...
                            [zeros(1, inMap); tmp1; zeros(1, inMap)] | ...
                            [zeros(2, inMap); tmp1];
                        tmp3(1:2,:) = tmp3(1:2,:) | tmp3(end-1:end,:);
                        tmp3(end-1:end,:) = [];
                        tmp4 = [tmp1; zeros(5, inMap)] | ...
                            [zeros(1, inMap); tmp1; zeros(4, inMap)] | ...
                            [zeros(4, inMap); tmp1; zeros(1, inMap)] | ...
                            [zeros(5, inMap); tmp1];
                        tmp4(1:5,:) = tmp4(1:5,:) | tmp4(end-4:end,:);
                        tmp4(end-4:end,:) = [];
                        tmp6 = [tmp1; zeros(8, inMap)] | ...
                            [zeros(1, inMap); tmp1; zeros(7, inMap)] | ...
                            [zeros(2, inMap); tmp1; zeros(6, inMap)] | ...
                            [zeros(6, inMap); tmp1; zeros(2, inMap)] | ...
                            [zeros(7, inMap); tmp1; zeros(1, inMap)] | ...
                            [zeros(8, inMap); tmp1];
                        tmp6(1:8,:) = tmp6(1:8,:) | tmp6(end-7:end,:);
                        tmp6(end-7:end,:) = [];
                        %layers{c}.Conn = [tmp3, tmp4];
                        if layers{c}.inDim(3) == layers{c}.outDim(3)
                            layers{c}.Conn = tmp6;
                        else
                            %layers{c}.Conn = [tmp4, tmp6];
                            layers{c}.Conn = rand(layers{c}.inDim(3), layers{c}.outDim(3)) > 0.66;
                        end
                    else
                        layers{c}.Conn = true(layers{c}.inDim(3), layers{c}.outDim(3));
                    end
                else
                    layers{c}.Conn = true(layers{c}.inDim(3), layers{c}.outDim(3));
                end

            case 'p'

                layers{c}.type = 'pool';
                poolDim = str2double(info(1:2));
                assert(~any(mod(layers{c-1}.outDim(1:2), poolDim)), 'poolDim must divide imageDim - filterDim + 1.');

                layers{c}.poolDim = poolDim;
                if numel(tmp) > 1
                    if strcmpi(tmp{2}, 'max') || strcmpi(tmp{2}, 'mean')
                        layers{c}.pooling = tmp{2};
                    else
                        layers{c}.pooling = 'mean';
                    end
                else
                    layers{c}.pooling = 'mean';
                end
                %layers{c}.avg_kern = ones(poolDim) / prod(poolDim);
                layers{c}.numFilters = layers{c-1}.outDim(3);
                layers{c}.inDim = layers{c-1}.outDim;
                layers{c}.outDim = layers{c-1}.outDim ./ [poolDim, 1];

            case 'f'

                layers{c}.type = 'fc';
                if c == 1
                    layers{c}.inDim = prod([numRows, numCols, numChannels]);
                else
                    layers{c}.inDim = prod(layers{c-1}.outDim);
                end
                
                if c == numel(layers) % if a current layer is the last, the number of output units is the number of classes.
                    layers{c}.outDim = numClasses;
                else % otherwise, the number of output units must be given.
                    assert(~isempty(info), 'the number of units in the fully-connected layer must be provided.');
                    layers{c}.outDim = str2double(info(1));
                end
                
                r  = sqrt(6) / sqrt(layers{c}.inDim + layers{c}.outDim + 1);
                layers{c}.W = rand(layers{c}.outDim, layers{c}.inDim) * 2 * r - r;
                layers{c}.b = zeros(layers{c}.outDim, 1);
                
                % parse dropout option
                layers{c}.dropout = 0;
                if numel(tmp) > 1
                    ttmp = regexp(tmp{2}, ':', 'split');
                    if strcmpi(ttmp{1}, 'dropout')
                    	layers{c}.dropout = str2double(ttmp{2});
                    end
                end
                
            case 'a'

                layers{c}.type = 'ae';
                if c == 1
                    layers{c}.inDim = prod([numRows, numCols, numChannels]);
                else
                    layers{c}.inDim = prod(layers{c-1}.outDim);
                end
                
                % the number of output units must be given.
                assert(~isempty(info), 'the number of hidden units must be provided.');
                layers{c}.outDim = str2double(info(1));
                
                r  = sqrt(6) / sqrt(layers{c}.inDim + layers{c}.outDim + 1);
                layers{c}.W1 = rand(layers{c}.outDim, layers{c}.inDim) * 2 * r - r;
                layers{c}.b1 = zeros(layers{c}.outDim, 1);
                layers{c}.W2 = rand(layers{c}.inDim, layers{c}.outDim) * 2 * r - r;
                layers{c}.b2 = zeros(layers{c}.inDim, 1);
                
                % parse dropout option
                layers{c}.dropout = 0;
                if numel(tmp) > 1
                    ttmp = regexp(tmp{2}, ':', 'split');
                    if strcmpi(ttmp{1}, 'dropout')
                    	layers{c}.dropout = str2double(ttmp{2});
                    end
                end
                
        end
    end