%% Programm for breath analysis:  data loading
clear all
%% Extract data from FTIR experiments
sdir='~/susmita/Susmita/Projects/BreathAnalysis/KlinikGrossAnalysis-2025/ALLDataGross/healthyCohort'; % Directory path
%filename=['20180125-06-KM-TB-102mb.6.dpt'; '20180125-07-KM-TB-202mb.7.dpt';'20180125-08-KM-TB-306mb.8.dpt';'20180125-09-KM-TB-371mb.9.dpt'];
%filename=['20181210-06-GLH.dpt'; '20181210-04-SLR.dpt'];
%filenames = [filename];
normVP = [420 420 428 448 417 430 420 449 483 499 ...
    438 465 438 428 503 505 504 454 515 441 ...
    404 363];
infoP = ["F" "M" "M" "F" "F" "F" "F" "M" "M" "M" ...
    "M" "F" "M" "M" "F" "M" "M" "M" "M" "M" ...
    "M" "M" ];

%files=dir([sdir '/*.dpt']);
files=dir('*.dpt');
[Nf Mf] = size(files)
dataM = [];
mol = [];
for i = 1:Nf%length(filenames)
    filename = files(i).name
    %filename = [sdir '/' files(i).name];
    %d = load([sdir files(i).name]); % load all data
     d = load(filename);
    idata = d(:,2);
    wn = d(:,1);                 % Wavenumber
    dataM = [dataM idata];        % Absorption from all experiments
    mol = {mol (files(i).name )};
end
save('Healthy.mat', 'files', 'infoP', 'normVP', 'dataM','wn');
%save('allHealthyData.mat', 'files', 'infoP', 'normVP', 'wn', 'dataM')