close all;
randomSeed = 12345;
rng(randomSeed);

datasetName = 'adult';
[dataAll, svarAll, groupNames] = loadData(datasetName);

seeds = 0:9;
k_values = 4:10;

bestOutOf = 10;
numIters = 10;

for ss = 1:length(seeds)
    seed = seeds(ss);
    rng(seed);  % for reproducibility

    for kk = 1:length(k_values)
        k = k_values(kk);

        % Load initial centers for the specific seed and k
        %centerFile = sprintf('initial_centers_seeds/adult_seed_%d_k_%d_centers.mat', seed, k);
        centerFile = sprintf('initial_centers_seeds/bank_married_single_seed_%d_k_%d_centers.mat', seed, k);
        %centerFile = sprintf('initial_centers_seeds/credit_MARRIAGE_seed_%d_k_%d_centers.mat', seed, k);

        %centerFile = sprintf('initial_centers_seeds/student_seed_%d_k_%d_centers.mat', seed, k);

        if ~isfile(centerFile)
            fprintf("Missing center file: %s\n", centerFile);
            continue;
        end
        S = load(centerFile);
        init_centers = S.centers;

        % Prepare the data
        data = dataAll;
        svar = svarAll +1;

        dataN = data;

        % Run unfair clustering
        [centersN, clusteringN, runtimeN, allCentersN, allCostsN] = lloyd(dataN, svar, k, numIters, bestOutOf, init_centers, 0, seed);

        % Run fair clustering
        [centersNF, clusteringNF, runtimeNF, allCentersNF, allCostsNF] = lloyd(dataN, svar, k, numIters, bestOutOf, init_centers, 1, seed);

        % Compute clustering costs
        costUnfair = compCost(data, svar, k, clusteringN, 0);
        costFair = compCost({data(svar == 1, :), data(svar == 2, :)}, svar, k, clusteringNF, 1);

        data_females = dataN(svar == 1, :);
        data_males   = dataN(svar == 2, :);

        % Create combined fair clustering labels
        labels_females = clusteringNF{1};
        labels_males = clusteringNF{2};
        fair_clustering_combined = zeros(size(data, 1), 1);
        fair_clustering_combined(svar == 1) = labels_females;
        fair_clustering_combined(svar == 2) = labels_males;
        disp("costUnfair"),disp(costUnfair)
        disp("costFair"),disp(costFair)

        % Save everything
        filename = sprintf('cost_seeds/full_results_seed_%d_k_%d_bank.mat', seed, k);
        save(filename, 'centersN', 'centersNF', 'clusteringN', 'clusteringNF', 'runtimeN', 'runtimeNF', ...
            'costUnfair', 'costFair', 'data_females', 'data_males', 'fair_clustering_combined', 'seed', 'k', ...
            'allCentersN', 'allCentersNF', 'allCostsN', 'allCostsNF', '-v7');

        fprintf("Saved results for seed=%d, k=%d\n", seed, k);
    end
end
