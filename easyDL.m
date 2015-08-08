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

if isnumeric(varargin{1}) % if the first argument is numeric, easyDL works in training mode.

    % the first four arguments are data, labels, model (or model signature), options
    data = varargin{1};
    labels = varargin{2};
    theta = varargin{3};
    options = varargin{4};
    if nargin > 4
        testdata = varargin{5};
        testlabels = varargin{6};
    end
    
    if size(data,4) == 1 % in case of "height x width x sample" format
        data = permute(data, [1 2 4 3]);
    end

    % the format of "data" is "height x width x channel x sample"
    [numRows, numCols, numChannels, numSamples] = size(data);

    % the format of "matlabels" is "class x sample"
    numClasses = max(labels);
    matlabels = zeros(numClasses, numSamples);
    for c = 1:numClasses, matlabels(c, labels==c) = 1; end

    % if a model signature is given, parse it.
    if ~isstruct(theta{1})
        layers = easyDLparseModel(theta, numRows, numCols, numChannels, numClasses);
    else % otherwise, use it.
        layers = theta;
    end

    % initialize incrementals
    inc = cell(numel(layers), 1);
    for c = 1:numel(layers)
        if strcmp(layers{c}.type, 'conv') || strcmp(layers{c}.type, 'fc')
            inc{c}.W = zeros(size(layers{c}.W));
            inc{c}.b = zeros(size(layers{c}.b));
        end
    end
    
    %%%%%%%%%%%%%%%
    %%% options %%%
    %%%%%%%%%%%%%%%
    
    % check required options
    assert(isfield(options, 'epochs'), 'the number of epochs must be provided.');
    
    % provide default options
    
    % set init learning rate to 0.1 and anneal it by factor of two after 10 epochs
    alpha = '0.1, 0.5@10';
    
    % set init momentum to 0.5 and change it to 0.95 after 20 iterations
    momentumList = {'0.5', '0.95@20'};
    
    % provide other default options
    minibatch = 100;
    weightDecay = 1e-4;
    sparseCoeff = 0;
    verbose = true;
    
    % get options from the input argument
    epochs = options.epochs;
    if isfield(options, 'alpha'),           alpha = options.alpha; end
    if isfield(options, 'minibatch'),       minibatch = options.minibatch; end
    if isfield(options, 'weightdecay'),     weightDecay = options.weightdecay; end
    if isfield(options, 'sparsecoeff'),     sparseCoeff = options.sparsecoeff; end
    if isfield(options, 'verbose'),         verbose = options.verbose; end
    
    % define a simple string parsing function
    getNumbers = @(str) str2double(regexp(str, '[,@]', 'split'));
    
    % parse learning rate information
    if ischar(alpha)
        alpha = getNumbers(alpha);
        if numel(alpha) > 1
            annealAlpha = alpha(2);
            annealAlphaEpoch = alpha(3);
            alpha = alpha(1);
        end
    end
    
    % parse a list of momentum values
    if iscell(momentumList)
        if ischar(momentumList{1})
            momentum = str2double(momentumList{1});
            momentumList(1) = [];
            if numel(momentumList) > 0
                tmp = getNumbers(momentumList{1});
                nextMomentum = tmp(1);
                nextMomentumIter = tmp(2);
            end
        else
            momentum = momentumList{1};
        end
    else
        momentum = momentumList(1);
    end
    
    iter = 0;

    for epoch = 1:epochs

        idx = randperm(numSamples);

        tic;

        for batch = 1:minibatch:(numSamples-minibatch+1)
            
            iter = iter + 1;
            
            % change momentum
            if numel(momentumList) > 0
                if iter == nextMomentumIter
                    momentum = nextMomentum;
                    if verbose
                        fprintf('Iter %d completed. Momentum is changed to %f.\n', iter, momentum);
                    end
                    momentumList(1) = [];
                    if numel(momentumList) > 0
                        tmp = getNumbers(momentumList{1});
                        nextMomentum = tmp(1);
                        nextMomentumIter = tmp(2);
                    end
                end
            end

            % get next randomly selected minibatch
            batchidx = idx(batch:batch+minibatch-1);
            clear('a');
            a = easyDLforward(layers, data(:,:,:,batchidx));
            target = matlabels(:,batchidx);
            M = size(target, 2);
            
            delta = cell(numel(layers)+1, 1);
            delta{end} = (a{end} - target) / M;% .* a{end} .* (1-a{end});
            
            for i = numel(layers):-1:1
                switch layers{i}.type
                case 'fc'

                    % calculate gradient
                    gradW = delta{i+1} * a{i}' + weightDecay * 2 * layers{i}.W;
                    inc{i}.W = momentum * inc{i}.W + alpha * gradW;
                    gradb = sum(delta{i+1}, 2);
                    inc{i}.b = momentum * inc{i}.b + alpha * gradb;
                    
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

                case 'pool'

                    delta{i} = zeros([layers{i}.inDim, M]);
                    for r = 1:layers{i}.poolDim(1)
                        for c = 1:layers{i}.poolDim(2)
                            delta{i}(r:layers{i}.poolDim(1):end,c:layers{i}.poolDim(2):end,:,:) = delta{i+1} / prod(layers{i}.poolDim);
                        end
                    end
                    delta(i+1) = [];

                case 'conv'

                    J = layers{i}.inDim(3);
                    K = layers{i}.outDim(3);
                    
                    tmpdelta = delta{i+1} .* a{i+1} .* (1-a{i+1});
                    for m = 1:M
                        for k = 1:K
                            tmpdelta(:,:,k,m) = rot90(tmpdelta(:,:,k,m),2);
                        end
                    end

                    gradW = zeros(size(layers{i}.W));
                    gradW_1norm = zeros(size(layers{i}.W));
                    for m = 1:M
                        for k = 1:K
                            for j = 1:J
                                gradW(:,:,j,k) = gradW(:,:,j,k) + conv2(a{i}(:,:,j,m), tmpdelta(:,:,k,m), 'valid');
                                %grad1W_1norm(:,:,filterNum) = grad1W_1norm(:,:,filterNum) + conv2(im(:,:,channelNum), deriv_a2(:,:,filterNum,imageNum), 'valid');
                                %bc_grad(filterNum) = bc_grad(filterNum) + sum(f_now(:));
                            end
                        end
                    end
                    
                    gradW = gradW ...
                        + weightDecay * layers{i}.W ...
                        + sparseCoeff * gradW_1norm;
                    inc{i}.W = momentum * inc{i}.W + alpha * gradW;
                    gradb = reshape(sum(sum(sum(tmpdelta,1),2),4),[],1);
                    inc{i}.b = momentum * inc{i}.b + alpha * gradb;
                    
                    if i > 1
                        delta{i} = zeros([layers{i}.inDim(1:2), J, M]);
                        for j = 1:J
                            for k = 1:K
                                tmpW = rot90(layers{i}.W(:,:,j,k),2);
                                for m = 1:M
                                    delta{i}(:,:,j,m) = delta{i}(:,:,j,m) + conv2(delta{i+1}(:,:,k,m), tmpW);
                                end
                            end
                        end
                        clear tmpW;
                    end
                    
                end
                
            end
            clear delta;

            for i = 1:numel(layers)
                if ~strcmp(layers{i}.type, 'pool')
                    layers{i}.W = layers{i}.W - inc{i}.W;
                    layers{i}.b = layers{i}.b - inc{i}.b;
                end
            end
            
        end

        ttt = toc;

        if verbose
            fprintf('Epoch %d completed.', epoch);
            % check with test dataset after each epoch
            if nargin > 4
                clear('a');
                a = easyDLforward(layers, testdata);
                [~, predlabels] = max(a{end}, [], 1);
                fprintf(' Test accuracy is %f.', sum(predlabels'==testlabels) / length(testlabels));
            end
            fprintf(' (%f sec)\n', ttt);
        end
        
        % anneal learning rate
        if exist('annealAlphaEpoch', 'var')
            if mod(epoch, annealAlphaEpoch) == 0
                alpha = alpha * annealAlpha;
                if verbose
                    fprintf('Learning rate is diminished to %f.\n', alpha);
                end
            end
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
    else % if the layer index is omitted, easyDL returns the last layer's activations.
        L = numel(layers) + 1;
    end
    
    clear('a');
    a = easyDLforward(layers, testdata);
    
    if L == numel(layers) + 1
        [~,out] = max(a{end},[],1);
        out = out';
    else
        out = a{L};
    end
    
end