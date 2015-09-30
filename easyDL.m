function out = easyDL(varargin)
%EASYDL Easy and fast deep learning.
%   EASYDL works in two different modes.
%
%   1. training
%   model = EASYDL(data, labels, model or model signature, options)
%      trains a supervised deep neural network with data-label pairs
%      and returns the model.
%   model = EASYDL(data, labels, model or model signature, options, testdata, testlabels)
%      works in the same manner except that the testing is performed after each epoch.
%
%   2. testing
%   output = EASYDL(model, testdata, n)
%      returns feed-forward values of testdata on the n-th layer in the model.
%      if the n is omitted, output is the last layer's activations.
%
%   Example (CNN with MNIST):
%
%       images : (28 x 28 x 1 x 60000) matrix which can be downloaded from the MNIST database and reshaped easily.
%       labels : (60000 x 1) vector which is pair information for images and ranges from 1 to 10.
%       testImages : (28 x 28 x 1 x 10000) matrix.
%       testLabels : (10000 x 1) vector.
%
%       clear('options');
%       options.epochs = 3;
%       options.weightdecay = 1e-5;
%
%       cnn = easyDL(images, labels, {'C:12@9x9', 'P:2x2', 'F'}, options);
%       predLabels = easyDL(cnn, testImages);
%       acc = sum(predLabels==testLabels) / length(predLabels);

%   Copyright (c) 2015 Taehoon Lee

% declare a simple string parsing function
global getNumbers;

% define a simple string parsing function
getNumbers = @(str) str2double(regexp(str, '[,@]', 'split'));

% declare convolutional functions
global forward_conv;
global backward_conv_weight;
global backward_conv_delta;

% define convolutional functions
forward_conv = @( a, W, b, conn, convidx, outsize ) ...
    permute( ...
        bsxfun(@plus, ...
            reshape( a(convidx)' * reshape(W, [], size(W, 4)), [outsize, size(W, 4)] ), ...
            reshape(b, 1, 1, 1, [])), ...
        [1 2 4 3] );

backward_conv_weight = @( a, delta, convidx, outsize ) ...
    reshape( ...
        a(convidx) * reshape(permute(delta, [1 2 4 3]), [], size(delta, 3)), ...
        outsize );

backward_conv_delta = @( delta_top, weight, convidx, outsize ) ...
     permute( ...
        reshape( delta_top(convidx)' * reshape(permute(weight, [1 2 4 3]), [], outsize(3)), ...
            [outsize(1:2), outsize(4), outsize(3)] ), ...
        [1 2 4 3] );

if isnumeric(varargin{1}) % if the first argument is numeric, easyDL works in training mode.

    % the first four arguments are data, labels, model (or model signature), options
    data = varargin{1};
    labels = varargin{2};
    theta = varargin{3};
    options = varargin{4};
    if nargin > 4,  testdata = varargin{5};     end
    if nargin > 5,  testlabels = varargin{6};   end
    
    % in case of "height x width x sample" format
    if size(data,4) == 1
        data = permute(data, [1 2 4 3]);
    end

    % the format of "data" is "height x width x channel x sample"
    [numRows, numCols, numChannels, numSamples] = size(data);

    % the format of "matlabels" is "class x sample"
    if ~isempty(labels)
        numClasses = max(labels);
        matlabels = zeros(numClasses, numSamples);
        for c = 1:numClasses, matlabels(c, labels==c) = 1; end
    end
    
    % if a model signature is given, parse it.
    if ~isstruct(theta{1})
        if ~isempty(labels)
            layers = easyDLparseModel(theta, numRows, numCols, numChannels, numClasses);
        else
            layers = easyDLparseModel(theta, numRows, numCols, numChannels);
        end
    else % otherwise, use it.
        layers = theta;
    end

    % parse options
    o = easyDLparseOptions(options);
    
    % index for convolution
    for c = 1:numel(layers)
        if strcmp(layers{c}.type, 'conv')
            K = layers{c}.outDim(3);
            M = o.minibatch;
            filterDim = [size(layers{c}.W, 1), size(layers{c}.W, 2)];
            layers{c}.ConvIdx = easyDLim2col([layers{c}.inDim, M], filterDim, false);
            layers{c}.BackConvIdx = easyDLim2col([layers{c}.outDim(1:2) + 2*filterDim - 2, K, M], filterDim, true);
            layers{c}.vpadding = zeros(filterDim(1)-1, layers{c}.outDim(2), K, M);
            layers{c}.hpadding = zeros(layers{c}.outDim(1) + 2*filterDim(1) - 2, filterDim(2)-1, K, M);
        end
    end
    
    % initialize incrementals
    inc = cell(numel(layers), 1);
    numParameters = 0;
    for c = 1:numel(layers)
        if strcmp(layers{c}.type, 'conv') || strcmp(layers{c}.type, 'fc')
            inc{c}.W = zeros(size(layers{c}.W));
            inc{c}.b = zeros(size(layers{c}.b));
            numParameters = numParameters + numel(layers{c}.b);
            if strcmp(layers{c}.type, 'fc')
                numParameters = numParameters + numel(layers{c}.W);
            elseif strcmp(layers{c}.type, 'conv')
                numParameters = numParameters + numel(layers{c}.W) * sum(layers{c}.Conn(:)) / numel(layers{c}.Conn);
            end
        elseif strcmp(layers{c}.type, 'ae')
            inc{c}.W1 = zeros(size(layers{c}.W1));
            inc{c}.b1 = zeros(size(layers{c}.b1));
            inc{c}.W2 = zeros(size(layers{c}.W2));
            inc{c}.b2 = zeros(size(layers{c}.b2));
            numParameters = numParameters + numel(layers{c}.W1) + numel(layers{c}.b1);
            numParameters = numParameters + numel(layers{c}.W2) + numel(layers{c}.b2);
        end
    end
    
    if o.gpu
        for c = 1:numel(layers)
            if strcmp(layers{c}.type, 'conv') || strcmp(layers{c}.type, 'fc')
                inc{c}.W = gpuArray(inc{c}.W);
                inc{c}.b = gpuArray(inc{c}.b);
                layers{c}.W = gpuArray(layers{c}.W);
                layers{c}.b = gpuArray(layers{c}.b);
            elseif strcmp(layers{c}.type, 'ae')
                inc{c}.W1 = gpuArray(inc{c}.W1);
                inc{c}.b1 = gpuArray(inc{c}.b1);
                inc{c}.W2 = gpuArray(inc{c}.W2);
                inc{c}.b2 = gpuArray(inc{c}.b2);
                layers{c}.W1 = gpuArray(layers{c}.W1);
                layers{c}.b1 = gpuArray(layers{c}.b1);
                layers{c}.W2 = gpuArray(layers{c}.W2);
                layers{c}.b2 = gpuArray(layers{c}.b2);
            end
        end
        data = gpuArray(data);
        matlabels = gpuArray(matlabels);
    end
    
    % show the number of parameters
    if o.verbose
        fprintf('The number of parameters is %d.\n', numParameters);
    end
    
    if strcmp(layers{1}.type, 'fc') || strcmp(layers{1}.type, 'ae')
        data = reshape(data, prod(layers{1}.inDim), []);
        if nargin > 4
            testdata = reshape(testdata, prod(layers{1}.inDim), []);
        end
    end
    
    iter = 0;
    if ndims(data) == 4
        isImageform = true;
    else
        isImageform = false;
    end

    for epoch = 1:o.epochs

        idx = randperm(numSamples);

        epochtime = tic;

        for batch = 1:o.minibatch:(numSamples-o.minibatch+1)
            
            iter = iter + 1;
            
            itertime = tic;
            
            % change momentum
            if numel(o.momentumList) > 0
                if iter == o.nextMomentumIter
                    o.momentum = o.nextMomentum;
                    if o.verbose
                        fprintf('Iter %d completed. Momentum is changed to %f.\n', iter, o.momentum);
                    end
                    o.momentumList(1) = [];
                    if numel(o.momentumList) > 0
                        tmp = getNumbers(o.momentumList{1});
                        o.nextMomentum = tmp(1);
                        o.nextMomentumIter = tmp(2);
                    end
                end
            end

            % get next randomly selected minibatch
            batchidx = idx(batch:batch+o.minibatch-1);
            M = numel(batchidx);
            
            for i = 1:numel(layers)
                if isfield(layers{i}, 'dropout') && layers{i}.dropout > 0
                    layers{i}.mask = ( rand([layers{i}.inDim, M]) > layers{i}.dropout ) ./ (1 - layers{i}.dropout);
                else
                    layers{i}.mask = [];
                end
            end
            
            if isImageform
                minibatch = data(:,:,:,batchidx);
                if o.rotation
                    minibatch = imrotate(minibatch, randi(21)-11, 'bilinear', 'crop');
                end
                if o.scaling
                    if rand < 0.5
                        minibatch = imresize(minibatch(3:26,3:26,:,:), [28 28]);
                    end
                end
            else
                minibatch = data(:,batchidx);
            end
            a = easyDLforward(layers, minibatch);
            
            % if supervised training
            if ~strcmp(layers{1}.type, 'ae')
                grad = easyDLbackward(layers, a, matlabels(:,batchidx), o);
                if o.adversarial || o.manifold
                    gradX = easyDLbackward_adv(layers, a, matlabels(:,batchidx));
                    if isImageform
                        %x_adv = a{1} - ( 1 - (epoch - 1) / o.epochs ) * bsxfun(@rdivide,gradX,sqrt(sum(sum(gradX.^2,1),2)));
                        x_adv = a{1} - bsxfun(@rdivide,gradX,sqrt(sum(sum(gradX.^2,1),2)));
                    else
                        %x_adv = a{1} - ( 1 - (epoch - 1) / o.epochs ) * bsxfun(@rdivide,gradX,sqrt(sum(gradX.^2,1)));
                        x_adv = a{1} - bsxfun(@rdivide,gradX,sqrt(sum(gradX.^2,1)));
                    end
                    x_adv(x_adv<0) = 0; x_adv(x_adv>1) = 1;
                    %if epoch>1, figure;colormap gray;imagesc(x_adv(:,:,1,1)); end
                    a_adv = easyDLfrward(layers, x_adv, false);
                    if o.adversarial
                        grad2 = easyDLbackward(layers, a_adv, matlabels(:,batchidx), o);
                        for i = 1:numel(layers)
                            if strcmp(layers{i}.type, 'conv') || strcmp(layers{i}.type, 'fc')
                                grad{i}.W = ( grad{i}.W + grad2{i}.W ) / 2;
                                grad{i}.b = ( grad{i}.b + grad2{i}.b ) / 2;
                            end
                        end
                    elseif o.manifold
                        grad2 = easyDLbackward2(layers, a, a_adv, matlabels(:,batchidx), o);
                        for i = 1:numel(layers)
                            if strcmp(layers{i}.type, 'conv') || strcmp(layers{i}.type, 'fc')
                                grad{i}.W = grad{i}.W + grad2{i}.W;
                                grad{i}.b = grad{i}.b + grad2{i}.b;
                            end
                        end
                    end
                    clear('a_adv', 'grad2');
                end
                for i = 1:numel(layers)
                    if strcmp(layers{i}.type, 'conv') || strcmp(layers{i}.type, 'fc')
                        inc{i}.W = o.momentum * inc{i}.W + o.alpha * grad{i}.W;
                        inc{i}.b = o.momentum * inc{i}.b + o.alpha * grad{i}.b;
                    end
                end
            % unsupervised
            else
                
                delta = cell(numel(layers)+1, 1);
                delta{2} = (a{2}{2} - a{1}) / M;% .* a{end} .* (1-a{end});
                delta{2}(a{2}{2}<0) = 0;
                
                % calculate gradient
                gradW = delta{2} * a{2}{1}' + o.weightdecay * layers{1}.W2;
                inc{1}.W2 = o.momentum * inc{1}.W2 + o.alpha * gradW;
                gradb = sum(delta{2}, 2);
                inc{1}.b2 = o.momentum * inc{1}.b2 + o.alpha * gradb;

                % update delta
                delta{1} = layers{1}.W2' * delta{2};
                delta{1} = delta{1} .* a{2}{1} .* (1-a{2}{1});
                delta(2) = [];

                % calculate gradient
                gradW = delta{1} * a{1}' + o.weightdecay * layers{1}.W1;
                inc{1}.W1 = o.momentum * inc{1}.W1 + o.alpha * gradW;
                gradb = sum(delta{1}, 2);
                inc{1}.b1 = o.momentum * inc{1}.b1 + o.alpha * gradb;

            end
            
            clear('a', 'delta');
            
            for i = 1:numel(layers)
                layers{i}.mask = [];
                if strcmp(layers{i}.type, 'conv') || strcmp(layers{i}.type, 'fc')
                    layers{i}.W = layers{i}.W - inc{i}.W;
                    layers{i}.b = layers{i}.b - inc{i}.b;
                elseif strcmp(layers{i}.type, 'ae')
                    layers{i}.W1 = layers{i}.W1 - inc{i}.W1;
                    layers{i}.b1 = layers{i}.b1 - inc{i}.b1;
                    layers{i}.W2 = layers{i}.W2 - inc{i}.W2;
                    layers{i}.b2 = layers{i}.b2 - inc{i}.b2;
                end
            end
            
            tt = toc(itertime);
            %disp(tt);
            
        end

        ttt = toc(epochtime);

        if o.verbose
            fprintf('Epoch %d completed.', epoch);
            % check with test dataset after each epoch
            if nargin > 4
                a = easyDLforward(layers, testdata);
                if strcmp(layers{1}.type, 'ae')
                    fprintf(' Test recon error is %f.', sqrt(mean(mean((a{end}{end} - a{1}).^2, 2), 1)));
                else
                    [~, predlabels] = max(a{end}, [], 1);
                    fprintf(' Test accuracy is %f.', sum(predlabels'==testlabels) / length(testlabels));
                end
                clear('a');
            end
            fprintf(' (%f sec)\n', ttt);
        end
        
        % anneal learning rate
        if isfield(o, 'annealAlphaEpoch')
            if mod(epoch, o.annealAlphaEpoch) == 0
                o.alpha = o.alpha * o.annealAlpha;
                if o.verbose
                    fprintf('Learning rate is diminished to %f.\n', o.alpha);
                end
            end
        end

    end
    
    if o.gpu
        for c = 1:numel(layers)
            if strcmp(layers{c}.type, 'conv') || strcmp(layers{c}.type, 'fc')
                layers{c}.W = gather(layers{c}.W);
                layers{c}.b = gather(layers{c}.b);
            elseif strcmp(layers{c}.type, 'ae')
                layers{c}.W1 = gather(layers{c}.W1);
                layers{c}.b1 = gather(layers{c}.b1);
                layers{c}.W2 = gather(layers{c}.W2);
                layers{c}.b2 = gather(layers{c}.b2);
            end
        end
    end
    
    for c = 1:numel(layers)
        if strcmp(layers{c}.type, 'conv')
            layers{c} = rmfield(layers{c}, {'ConvIdx', 'BackConvIdx', 'vpadding', 'hpadding'});
        end
    end

    out = layers;

elseif iscell(varargin{1}) % if the first argument is cell-type, easyDL works in testing mode.
    
    % the first two arguments are model and testdata
    layers = varargin{1};
    testdata = varargin{2};

    if size(testdata,4) == 1 % in case of "height x width x sample" format
        testdata = permute(testdata, [1 2 4 3]);
    end
    
    % the third argument is an layer index and optional
    if nargin > 2
        L = varargin{3};
    else % if the layer index is omitted, easyDL returns the predicted labels.
        L = numel(layers) + 2;
    end
    
    if strcmp(layers{1}.type, 'fc') || strcmp(layers{1}.type, 'ae')
        testdata = reshape(testdata, prod(layers{1}.inDim), []);
    end
    
    a = easyDLforward(layers, testdata);
    
    if L == numel(layers) + 2
        if iscell(a{end})
            out = a{end};
        else
            [~,out] = max(a{end},[],1);
            out = out';
        end
    elseif L == 0
        out = a;
    else
        out = a{L};
    end
    
    clear('a');
    
end