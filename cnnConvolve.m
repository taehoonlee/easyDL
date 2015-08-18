function convolvedFeatures = cnnConvolve(images, W, b, Conn)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

[imageR, imageC, ~, M] = size(images);
[filterR, filterC, J, K] = size(W);
convR = imageR - filterR + 1;
convC = imageC - filterC + 1;

convolvedFeatures = zeros(convR, convC, K, M);

filter = rot90(W,2);

if true
    for k = 1:K
        if J > 1
            for j = 1:J
                if Conn(j,k)
                    convolvedFeatures(:,:,k,:) = convolvedFeatures(:,:,k,:) + convn(images(:,:,j,:), filter(:,:,j,k), 'valid');
                end
            end
        else
            for m = 1:M
                convolvedFeatures(:,:,k,m) = convolvedFeatures(:,:,k,m) + conv2(images(:,:,1,m), filter(:,:,1,k), 'valid');
            end
        end
        convolvedFeatures(:,:,k,:) = convolvedFeatures(:,:,k,:) + b(k);
    end
else
    for filterNum = 1:numFilters
        convolvedFeatures(:, :, filterNum, :) = sum(convn(images, filter(:,:,filterNum), 'valid'),3) + b(filterNum);
    end
end

%convolvedFeatures = bsxfun(@plus, convolvedFeatures, permute(b,[2 3 1]));

%---------------------
%convolvedFeatures(convolvedFeatures<=0) = convolvedFeatures(convolvedFeatures<=0) / 2;
convolvedFeatures = 1./(1+exp(-convolvedFeatures));

end