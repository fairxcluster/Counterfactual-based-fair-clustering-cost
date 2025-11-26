function [dataAll, svarAll, groupNames] = loadData(datasetName)

    if strcmp(datasetName, 'adult')

        %dataAll = csvread('adult_preprocessed',1,0);  % make sure this file exists svarAll = dataAll(:, 12);dataAll = dataAll(:, 1:10);
        dataAll = csvread('student_preprocessed.csv',1,0) %student dataset svarAll = dataAll(:, 28); dataAll = dataAll(:, 1:27);
        %dataAll = csvread('credit_preprocessed.csv',1,0) %credit dataset svarAll = dataAll(:, 16); dataAll = dataAll(:, 1:15);
        %dataAll = csvread('bank_preprocessed.csv',1,0) %bank dataset svarAll = dataAll(:, 4);dataAll = dataAll(:, 1:3);
        svarAll = dataAll(:, 28);              % assuming column 10 is the sensitive attribute
        dataAll = dataAll(:, 1:27);
        disp('unique')
        disp(unique(svarAll))      

        groupNames = {'Female'; 'Male'};
    else
        error('Unsupported dataset');
    end
end


