close all;
clc;
clear all;

addpath('../lib');
addpath('../bin');
addpath('../tests');

numpts = 1e4;

x0 = -1;
y0 = 0;

xf = 1;
yf = 0;

% fix the seed
rng(1);

% 
dataTable = table('Size',[numpts, 9], 'VariableTypes', {'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'}, 'VariableNames', {'Pi_x', 'Pi_y', 'Pi_th', 'Pm_x', 'Pm_y', 'Pf_x', 'Pf_y', 'Pf_th', 'kmax'});

xm_interval = [-3,3];
ym_interval = [-3,3];
thi_interval = [-pi, pi];
thf_interval = [-pi, pi];
kmax_interval = [0.2, 2];

for i =1:numpts
  if mod(i, 1e4) == 0
    fprintf('i = %d\n', i);
  end
  dataTable.Pi_x(i) = x0;
  dataTable.Pi_y(i) = y0;
  dataTable.Pf_x(i) = xf;
  dataTable.Pf_y(i) = yf;

  r = rand();

  dataTable.Pi_th(i) = thi_interval(1)  + (thi_interval(2)  - thi_interval(1))*r;
  dataTable.Pm_x(i)  = xm_interval(1)   + (xm_interval(2)   - xm_interval(1))*r;
  dataTable.Pm_y(i)  = ym_interval(1)   + (ym_interval(2)   - ym_interval(1))*r;
  dataTable.Pf_th(i) = thf_interval(1)  + (thf_interval(2)  - thf_interval(1))*r;
  dataTable.kmax(i)  = kmax_interval(1) + (kmax_interval(2) - kmax_interval(1))*r;


end

save('store_possible_combination.mat', 'dataTable');

function res = unifSample(interval)
  res = interval(1) + (interval(2) - interval(1))*rand();
end