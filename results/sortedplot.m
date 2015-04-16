function sortedplot( data, x, y, numRep, var, spec )
%SORTEDPLOT Summary of this function goes here
%   Detailed explanation goes here

    SEM = sqrt(data(:, var)) ./ sqrt(data(:, numRep)); % Standard Error
    ts = tinv(0.005, data(:, numRep)-1);      % T-Score
    CI = ts .* SEM;
    
    [x, I] = sort(data(:, x));
    errorbar(x, data(I, y), CI, -CI, spec);

end

