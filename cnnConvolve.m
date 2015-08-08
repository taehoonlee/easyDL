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

[imageR, imageC, ~, numImages] = size(images);
[filterR, filterC, numInFilters, numOutFilters] = size(W);
convR = imageR - filterR + 1;
convC = imageC - filterC + 1;

convolvedFeatures = zeros(convR, convC, numOutFilters, numImages);

filter = zeros(size(W));
for outFilterNum = 1:numOutFilters
    for inFilterNum = 1:numInFilters
        filter(:,:,inFilterNum,outFilterNum) = rot90(W(:,:,inFilterNum,outFilterNum),2);
    end
end

if true
    for imageNum = 1:numImages
        for outFilterNum = 1:numOutFilters
            for inFilterNum = 1:numInFilters
                if Conn(inFilterNum,outFilterNum)
                    convolvedFeatures(:, :, outFilterNum, imageNum) = convolvedFeatures(:, :, outFilterNum, imageNum) + ...
                        conv2(images(:,:,inFilterNum,imageNum),filter(:,:,inFilterNum,outFilterNum),'valid');
                end
            end
            %convolvedFeatures(:, :, filterNum, imageNum) = conv2(images(:,:,filterNum,imageNum),filter(:,:,filterNum),'valid');
            convolvedFeatures(:, :, outFilterNum, imageNum) = convolvedFeatures(:, :, outFilterNum, imageNum) + b(outFilterNum);
        end
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