%% Programm for breath analysis:  data loading
clear all
%% Extract data from FTIR experiments
sdir='~/mpqLMU/breathAnalysis/dataBA2019/kgAnalysis/allKgData/'; % Directory path
%filename=['20180125-06-KM-TB-102mb.6.dpt'; '20180125-07-KM-TB-202mb.7.dpt';'20180125-08-KM-TB-306mb.8.dpt';'20180125-09-KM-TB-371mb.9.dpt'];
%filename=['20181210-06-GLH.dpt'; '20181210-04-SLR.dpt'];
%filenames = [filename];
normVP = [504 425 451 454 450 474 451 471 540 467 ...
    550 468 481 450 515 441 452 462 453 450 452 ...
    490 504 520 525 498 542 527 550];
infoP = ["H" "PC" "PC" "H" "PC" "BC" "PC" "PC" "BC" "BC" ...
    "PC" "PC" "PC" "PC" "H" "H" "PC" "PC" "KC" "PC" "KC" ...
    "PC" "BC" "BC" "PC" "KC" "PC" "PC" "PC"];
files=dir('*.dpt');
[Nf Mf] = size(files)
dataM = [];
mol = [];
for i = 1:Nf%length(filenames)
    d = load([sdir files(i).name]); % load all data
    idata = d(:,2);
    wn = d(:,1);                 % Wavenumber
    dataM = [dataM idata];        % Absorption from all experiments
    mol = {mol (files(i).name )};
end
save('allKGdata.mat', 'files', 'infoP', 'normVP', 'wn', 'dataM')