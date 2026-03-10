clear all
L = load('kgDataBCNew.mat')
normF = [474 540 467 504 520]; % first 10 for KG sample and last 
                                % three arr ML, EF and OL
%% First order Baseline correction: this only shift the base line after CO2 to 0
dataR = L.dataM;
wn = L.wn;
files = L.files;
[Mf Nf] = size(dataR);

figure;

plot(wn, dataR,'LineWidth',4);
set(gca,'FontSize',40,'LineWidth',4.25);
%xrange ([900 1020])
xlabel('\fontsize{40} Wavenumbers in cm^{-1}')
ylabel('\fontsize{40} Absorbance')
ylim([-0.001 0.006])
hold on;
refline([0 0])
figure;
plot(wn, dataR(:,2))


%indx = find(wn(:) >2540 & wn(:,1)<2560);
indx = find(wn(:) >2550 & wn(:,1)<2600);
offSet = [];
for i = 1:Nf%length(filenames)
    tot = 0;
    for j =1:length(indx)
        tot = tot + dataR(indx(j),i);
    end
    offset = tot/length(indx);
    offSet = [offSet offset];
end 
data = [];
for i = 1:Nf%length(filenames)
    dd = dataR(:,i)-offSet(i);
    data = [data dd];
end


%% Select the spectral region
 mf = 1005 %3000;
 limL = 990; %2880;
 limH = 1020; %3200;

indL = find(wn(:) >limL & wn(:,1)<limH);
dataL = [];
for i = 1:length(indL)
    d = data(indL(i),:);
    wl = wn(indL(i));
    dL = [wl d];
    dataL = [dataL; dL];
end



% % %% PNNL data extraction from .TXT file. Automatic upload all molecular data from PNNL
% % pdir='~/mpqLMU/specDataBase/specT25pnnl/'; %PNNL spectral directory
% % filePNNL = dir([pdir '*.TXT']);  % Read all PNNL data file name with .TXT
% % pnnlMol = {filePNNL.name}; %List of PNNL molecules
% % xq = limL:1:limH;    % equispaced data point at each wavenumber
% % pnnlI = [];
% % for i = 1:length(filePNNL)
% %     dataPNNL = load([pdir filePNNL(i).name]); % load all data for each molecule
% %     indPNNL = find(dataPNNL(:,1)>limL &dataPNNL(:,1)<limH); % Find the index of the data in mentioned limit
% %     pnnl = [];
% %     for i=1:length(indPNNL)
% %         pwn = [dataPNNL(indPNNL(i),1), dataPNNL(indPNNL(i),2)]; % Pickup the data points in the limit
% %         pnnl = [pnnl; pwn];
% %     end
% %     pnnlInt = interp1(pnnl(:,1), pnnl(:,2), xq,'spline'); % Interpolate to equidistance data point
% %     pnnlI = [pnnlI pnnlInt'];
% % end
%% Second order baseline correction
dataS = [];
for i = 1:Nf%length(filenames)
    bcf = dataL(end,i+1):(dataL(1,i+1)-dataL(end,i+1))/length(dataL):dataL(1,i+1);
    bcf(end) = [];
    data = dataL(:,i+1) + bcf';
    data = data - data(1);
    data = data*(500/normF(i));
    dataS = [dataS data];        % Absorption from all experiments
end

%% Averaging of data
tData =  dataS';
avData = mean(tData);
avData = avData';
avData = smooth(avData);

%% Saving segment of data

wN = dataL(:,1); 
Data = [dataL(:,1) avData];

save(['AvDatakgAvBC_25' num2str(mf) '.mat'], 'wN', 'avData')
dataText = fopen(['AvDatakgAvBC_25' num2str(mf) '.txt'],'w');
 for i =1:size(Data,1)
     fprintf(dataText,'%g\t',Data(i,:));
     fprintf(dataText,'\n');
 end
     fclose(dataText)

%% Ploting data

for i = 1:length(files)
    figure(i)
    clf
    plot(dataL(:,1),(dataS(:,i)),'b','LineWidth',4)
    hold all
    plot(dataL(:,1),(avData),'r','LineWidth',4)
    set(gca,'FontSize',40,'LineWidth',4.25)
    %legend(files(i).name(1:end-4))
    %set(legend,'Position',[0.5,0.78752,0.2,0.1],'linewidth',2);
    %legend boxoff
    %title('Smoking effect')
    xlabel('\fontsize{40} Wavenumbers in cm^{-1}')
    ylabel('\fontsize{40} Absorbance')
    title(['Case' files(i).name(end-7:end-4)])
    %ylim([-0.00151 0.004])
    %xlim([720 1640])
    xlim([limL limH])
end


%for i = 1:length(files)
    figure(111)
    clf
    plot(dataL(:,1),(avData),'r','LineWidth',4)
    set(gca,'FontSize',40,'LineWidth',4.25)
    %legend(files(i).name(1:end-4))
    %set(legend,'Position',[0.5,0.78752,0.2,0.1],'linewidth',2);
    %legend boxoff
    %title('Smoking effect')
    xlabel('\fontsize{40} Wavenumbers in cm^{-1}')
    ylabel('\fontsize{40} Absorbance')
    title(['Case' files(i).name(end-7:end-4)])
    %ylim([-0.00151 0.004])
    %xlim([720 1640])
    xlim([limL limH])
%end

figure;
plot(dataL(:,1),avData,'r','LineWidth',4);
set(gca,'FontSize',40,'LineWidth',4.25)
 xlabel('\fontsize{40} Wavenumbers in cm^{-1}')
    ylabel('\fontsize{40} Absorbance')
    title(['Average of BC']);
    xlim([limL limH])


colors = jet(10);
    figure(555)
    clf
for i = 1:length(files)

    plot(dataL(:,1),(dataS(:,i)-dataS(:,1)),'Color',colors(i,:),'LineWidth',4)
    %plot(dataL(:,1),(dataS(:,i)-dataS(:,end)),'Color',colors(i,:),'LineWidth',4)
    hold all
    set(gca,'FontSize',30,'LineWidth',4.25)
    legendInfo{i}=[files(i).name(1:8) files(i).name(end-6:end-4)]; % or whatever is appropriate
    legend(legendInfo)
    %legend(i)
    %legend(files(i).name(13:end-4))
    set(legend,'Position',[0.175,0.642,0.2,0.1],'linewidth',2);
    legend boxoff
    %title(files(end).name(1:8))
    xlabel('\fontsize{40} Wavenumbers in cm^{-1}')
    ylabel('\fontsize{40} Absorbance')
    %ylim([-0.00151 0.004])
    %xlim([720 1640])
    xlim([limL limH])
end

%refL = refline([0 0]);
refL.Color = 'k';
refL.LineStyle = '--';
refL.LineWidth = 2;