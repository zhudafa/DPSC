function createfigure_Gaussian(X1, YMatrix1, dataset)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 18-Apr-2018 11:17:27

% Create figure
figure1 = figure('Color',[1 1 1]);

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'MarkerSize',8,'LineWidth',1.5,'LineStyle','-.','Parent',axes1);
set(plot1(1),'DisplayName','KNN','Marker','d','Color',[0.93 0.69 0.13]);
set(plot1(2),'DisplayName','EA','Marker','^', 'Color',[0 0.45 0.74]);
set(plot1(3),'DisplayName','TPG','Marker','*','Color',[0.47 0.67 0.19]);
set(plot1(4),'DisplayName','RDP','Marker','>','Color',[0 0 0]);
%set(plot1(5),'DisplayName','LPGMM','Marker','square','Color',[0.6 0.2 0.5]);
set(plot1(5),'DisplayName','SSD(ours)','Marker','o','Color',[1 0 0]);


% Create xlabel
xlabel('Number of clusters');

% Create title
title(dataset);

% Create ylabel
ylabel('Accuracy');

box(axes1,'on');
grid(axes1,'on');
set(axes1,'GridLineStyle','--');

% Create legend
legend(axes1,'show');

