function [centers, clustering, runtime, allCenters, allCosts, kmLoss, kmLossGroups] = ...
    lloyd_kmeans_loss(data, svar, k, numIters, bestOutOf, randCenters, isFair, seed)

    minCost = inf;
    runtime = 0;
    datasep = cell(1,2);
    allCenters = cell(bestOutOf, 1);
    allCosts = zeros(bestOutOf, 2);

    % defaults for new outputs (optional)
    kmLoss = NaN;
    kmLossGroups = [NaN NaN];

    if isFair == 1
        datasep{1} = data(svar == 1, :);
        datasep{2} = data(svar == 2, :);

        data = datasep;
        dataTemp = [data{1}; data{2}];
        ns = [size(data{1}, 1), size(data{2}, 1)];
    else
        dataTemp = data;
        ns = [1, 1];
    end

    for i = 1:bestOutOf

        currentCenters = randCenters;
        allCenters{i} = currentCenters;

        tStart = tic;

        for j = 1:numIters
            if j == 1 && isFair == 1
                % Load unfair clustering labels for female and male
                filename = sprintf('clustering_file_with_seed_lambda/clustering_adult_seed_%d_lambda_0.0.mat', seed);
                if isfile(filename)
                    loaded = load(filename);
                else
                    error('Unfair clustering file not found: %s', filename);
                end

                % Create clustering as a cell array (adult/student)
                currentClustering = cell(1, 2);
                currentClustering{1} = loaded.female_labels; % svar==1
                currentClustering{2} = loaded.male_labels;   % svar==2

                % Recompute centers from this clustering
                currentCenters = findCenters(data, svar, k, currentClustering, isFair);
                continue
            end

            if j == numIters
                currentClustering = findClustering(dataTemp, ns, currentCenters, 1, isFair);
            else
                currentClustering = findClustering(dataTemp, ns, currentCenters, 0, isFair);
                currentCenters    = findCenters(data, svar, k, currentClustering, isFair);
            end
        end

        runtime = runtime + toc(tStart);
        currentCost = compCost(data, svar, k, currentClustering, isFair);
        allCosts(i, :) = currentCost(:)';

        if i == 1 || currentCost < minCost
            minCost   = currentCost;
            centers   = currentCenters;
            clustering = currentClustering;
        end
    end

    runtime = runtime / bestOutOf;

    % -------- compute k-means loss for the final solution --------
    if isFair == 1
        % reconstruct full data and label vector (1..k)
        X = [data{1}; data{2}];
        a = zeros(size(X,1),1);
        a(1:ns(1))           = clustering{1};
        a(ns(1)+1:end)       = clustering{2};
        kmLoss               = kmeans_loss(X, centers, a);
        % group-wise
        kmLossGroups(1)      = kmeans_loss(data{1}, centers, clustering{1});
        kmLossGroups(2)      = kmeans_loss(data{2}, centers, clustering{2});
    else
        X = dataTemp;             % = original data
        a = clustering;           % N x 1 labels (1..k)
        kmLoss = kmeans_loss(X, centers, a);
        % group-wise (using svar)
        g1 = (svar == 1); g2 = (svar == 2);
        kmLossGroups(1) = kmeans_loss(X(g1,:), centers, a(g1));
        kmLossGroups(2) = kmeans_loss(X(g2,:), centers, a(g2));
    end
    % --------------------------------------------------------------
end
