function [centers, clustering, runtime, allCenters, allCosts] = ...
    lloyd(data, svar, k, numIters, bestOutOf, randCenters, isFair,seed)

    minCost = inf;
    runtime = 0;
    datasep = cell(1,2);
    allCenters = cell(bestOutOf, 1);
    allCosts = zeros(bestOutOf, 2);
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

        %Print based on clustering type
        if isFair == 1 && i == 1
            %disp("DATA TEMP"),disp(size(dataTemp))
            %disp('→ First centers from FAIR clustering:')
            %disp(currentCenters)
        end
        if isFair == 1 && i == 2
            %disp('→ Second centers from FAIR clustering:')
            %disp(currentCenters)
        end

        if isFair == 1 && i == bestOutOf
            %disp('→ Last centers from FAIR clustering:')
            %disp(currentCenters)
        end

        if isFair == 0 && i == bestOutOf
            %disp('→ Last centers from UNFAIR clustering:')
            %disp(currentCenters)
        end
        if isFair == 0 && i == 1
            %disp('→ First centers from UNFAIR clustering:')
            %disp(currentCenters)
        end

        tStart = tic;

        %disp("numIters"), disp(numIters)
        for j = 1:numIters
            if j == 1 && isFair == 1
                % Load unfair clustering labels for female and male
                filename = sprintf('clustering_file_with_seed_lambda/clustering_student_seed_%d_lambda_0.0.mat', seed);
                %filename = sprintf('clustering_file_with_seed_lambda/clustering_adult_seed_%d_lambda_0.0.mat', seed);
                %filename = sprintf('clustering_file_with_seed_lambda/clustering_bank_seed_%d_lambda_0.0.mat', seed);
                %filename = sprintf('clustering_file_with_seed_lambda/clustering_credit_seed_%d_lambda_0.0.mat', seed);


                %filename = sprintf('clustering_file_with_seed_lambda/unfair_clustering_bank_seed_%d_k_%d.mat', seed, k);
                %filename = sprintf('clustering_file_with_seed_lambda/unfair_clustering_credit_seed_%d_k_%d.mat', seed, k);
                %filename = sprintf('clustering_file_with_seed_lambda/unfair_clustering_student_seed_%d_k_%d.mat', seed, k);
                %filename = sprintf('clustering_file_with_seed/unfair_clustering_credit_seed_%d_k_%d.mat', seed, k);


                if isfile(filename)
                    loaded = load(filename);
                else
                    error('Unfair clustering file not found: %s', filename);
                end
                
                % Create clustering as a cell array

                %credit-bank 
                currentClustering = cell(1, 2);
                %currentClustering{1} = loaded.married_labels;
                %currentClustering{2} = loaded.single_labels;

                %adult-student
                currentClustering{1} = loaded.female_labels;
                currentClustering{2} = loaded.male_labels;


                % Compute centers from this clustering
                %disp(currentClustering) %take the unfair clustering brom balance
                %disp('DATA'),
                %disp(size(data{1})), disp(size(data{2}))
                %disp("currentClustering"),
                %disp(length(currentClustering{1})), disp(length(currentClustering{2}))

                currentCenters = findCenters(data, svar, k, currentClustering, isFair);
                %disp("INITIAL CENTERS FROM UNFAIR CLUSTERING"), disp(currentCenters)

                % Continue loop
                continue
            end

            if j == numIters
                %disp("currentCenters"), disp(currentCenters), %disp("IS FAIR"), %disp(isFair)

                currentClustering = findClustering(dataTemp, ns, currentCenters, 1, isFair);
                %disp(currentClustering)
            else
                

                currentClustering = findClustering(dataTemp, ns, currentCenters, 0, isFair);
                %disp("CURRENT CLUSTERING FROM LLOYD"), %disp(currentClustering)
                currentCenters = findCenters(data, svar, k, currentClustering, isFair);
                %disp(isFair), disp("CURRENT CENTERS FROM LLOYD"), %disp(j), %disp(currentCenters)
            end
        end

        runtime = runtime + toc(tStart);
        currentCost = compCost(data, svar, k, currentClustering, isFair);
        allCosts(i, :) = currentCost(:)';
        if i == 1 || currentCost < minCost
            minCost = currentCost;
            centers = currentCenters;
            clustering = currentClustering;
        end
    end

    runtime = runtime / bestOutOf;
end
