%% Programm for breath analysis:  data loading
clear all
%% Extract data from FTIR experiments
sdir='~/mpqLMU/breathAnalysis/dataBA2020/kg20200925/'; % Directory path
%filename=['20180125-06-KM-TB-102mb.6.dpt'; '20180125-07-KM-TB-202mb.7.dpt';'20180125-08-KM-TB-306mb.8.dpt';'20180125-09-KM-TB-371mb.9.dpt'];
%filename=['20181210-06-GLH.dpt'; '20181210-04-SLR.dpt'];
%filenames = [filename];
idPerson = ["KM" "AA" "BJ" "ÖG" "030" "031" "032" "033" "034" "035" ...
    "036" "037" "038" "039" "040" "041" "042" "043" "044" "045" "046" "047" "048"];
normVP = [505 503 478 453 460 494 410 413 479 489 ...
    473 464 445 499 406 455 481 388 428 466 463 ...
    520 461];
infoP = ["H" "H" "H" "H" "KG" "KG" "KG" "KG" "KG" "KG" "KG" "KG" "KG" ...
     "KG" "KG" "KG" "KG" "KG" "KG" "KG" "KG" "KG" "KG"];
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
save('kgdata20200925.mat', 'files', 'idPerson', 'infoP', 'normVP', 'wn', 'dataM')