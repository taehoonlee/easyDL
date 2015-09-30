function options = easyDLparseOptions(options)

    % check required options
    assert(isfield(options, 'epochs'), 'the number of epochs must be provided.');

    % provide default options

    % set init learning rate to 0.1 and anneal it by factor of two after 10 epochs
    if ~isfield(options, 'alpha'),          options.alpha = '0.1, 0.5@10'; end

    % set init momentum to 0.5 and change it to 0.95 after 20 iterations
    if ~isfield(options, 'momentumList'),   options.momentumList = {'0.5', '0.95@20'}; end

    % provide other default options
    if ~isfield(options, 'minibatch'),      options.minibatch = 100; end
    if ~isfield(options, 'weightdecay'),    options.weightdecay = 1e-4; end
    if ~isfield(options, 'sparsecoeff'),    options.sparsecoeff = 0; end
    if ~isfield(options, 'verbose'),        options.verbose = true; end
    if ~isfield(options, 'scaling'),        options.scaling = false; end
    if ~isfield(options, 'rotation'),       options.rotation = false; end
    if ~isfield(options, 'gpu'),            options.gpu = false; end
    
    % employ the simple string parsing function
    global getNumbers;
    
    % parse learning rate information
    if ischar(options.alpha)
        options.alpha = getNumbers(options.alpha);
        if numel(options.alpha) > 1
            options.annealAlpha = options.alpha(2);
            options.annealAlphaEpoch = options.alpha(3);
            options.alpha = options.alpha(1);
        end
    end

    % parse a list of momentum values
    if iscell(options.momentumList)
        if ischar(options.momentumList{1})
            options.momentum = str2double(options.momentumList{1});
            options.momentumList(1) = [];
            if numel(options.momentumList) > 0
                tmp = getNumbers(options.momentumList{1});
                options.nextMomentum = tmp(1);
                options.nextMomentumIter = tmp(2);
            end
        else
            options.momentum = options.momentumList{1};
        end
    else
        options.momentum = options.momentumList(1);
    end
    
    %%%%% advanced options
    if ~isfield(options, 'adversarial'),        options.adversarial = false; end
    if ~isfield(options, 'manifold'),           options.manifold = 0; end
