%=========================================================================%
%                                                                         %
%  Autors: Enrico Bertolazzi                                              %
%          Department of Industrial Engineering                           %
%          University of Trento                                           %
%          enrico.bertolazzi@unitn.it                                     %
%          m.fregox@gmail.com                                             %
%                                                                         %
%=========================================================================%
% Driver test program to check Clothoids lib                              %
%=========================================================================%

close all;
clc;
clear all;

addpath('../lib');
addpath('../bin');
addpath('../tests');

k_max  = 0.6;
d      = 3;
%
x0     = -d;
y0     = 0;
xM     = 0;
yM     = 0.5*d;
xf     = 2.1*d;
yf     = 0;
theta0 = -pi/2;
thetaf = 0.0;

flag_plot = false;

dataTable = read_SACON_file();

%%
if flag_plot
  figure();
end
res2store = {};

for i=1:size(dataTable,1)
  x0 = dataTable.Pi_x(i);
  y0 = dataTable.Pi_y(i);
  theta0 = dataTable.Pi_th(i);
  xM = dataTable.Pm_x(i);
  yM = dataTable.Pm_y(i);
  xf = dataTable.Pf_x(i);
  yf = dataTable.Pf_y(i);
  thetaf = dataTable.Pf_th(i);
  k_max = dataTable.kmax(i);
  print_saccon_data(dataTable,i);
  res2store{end+1} = simulateNplot( x0, y0, theta0, xM, yM, xf, yf, thetaf, k_max , flag_plot);
  if flag_plot
    pause(0.5)
  end
end

% save results to .mat file
save('results_saccon_compare.mat','res2store');

%%
% histogrem from res2store{:}{1}.type_word

types = repmat({''},1,size(dataTable,1));
typesN = zeros(1,size(dataTable,1));
for i=1:size(dataTable,1)
  types{i} = res2store{i}{1}.type_word;
  typesN(i) = word2number(types{i});
end




figure();
histogram(typesN);


%% from literlas to cases
% transform types from 'LSLLRL' to a number
function id = word2number(case_string)
  % \begin{tabular}{ll}
  %   CCCCCC & RLRRLR$_6$, LRLLRL$_6$\\
  %   CCCCSC & RLRRSR$_8$, RLRRSL$_{20}$, LRLSL$_8$, LRLLSR$_8$\\
  %   CSCCCC & RSRRLR$_8$, LSRRLR$_{20}$, RSLLRL$_{20}$, LSLLRL$_8$\\
  %   CSCCSC & RSRRSR$_4$, LSRRSR$_8$, RSRRSL$_8$, LSRRSL$_8$, LSLLSL$_4$, RSLLSL$_8$, LSLLSR$_8$, RSLLSR$_8$
  % \end{tabular}

  % 'RLRRLR', 'LRLLRL', 'RLRRSR', 'RLRRSL', 'LRLLSL', 'LRLLSR', 'RSRRLR', 'LSRRLR', 'RSLLRL', 'LSLLRL', 'RSRRSR', 'LSRRSR', 'RSRRSL', 'LSRRSL', 'LSLLSL', 'RSLLSL', 'LSLLSR', 'RSLLSR' 
  map = containers.Map({'RLRRLR', 'LRLLRL', 'RLRRSR', 'RLRRSL', 'LRLLSL', 'LRLLSR', 'RSRRLR', 'LSRRLR', 'RSLLRL', 'LSLLRL', 'RSRRSR', 'LSRRSR', 'RSRRSL', 'LSRRSL', 'LSLLSL', 'RSLLSL', 'LSLLSR', 'RSLLSR'},...
                        {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});

  % Use the map
  if isKey(map, case_string)
    id = map(case_string);
  else
    id = -1;
  end

end


%% test with conditions
function results = simulateNplot( x0, y0, theta0, xM, yM, xf, yf, thetaf, k_max,flag_plot)

  DB_A = Dubins();
  DB_B = Dubins();
  DB3  = Dubins3p();
  DB3.set_tolerance(pi/180/100);

  len    = [];
  DlenA  = [];
  DlenB  = [];
  kind   = [];
  epsilon = pi/180/100;

  % 'pattern'

  tic();
  flag = DB3.build( x0, y0, theta0, xM, yM, xf, yf, thetaf, k_max, 'pattern' );
  elapsed = toc();
  DB3_data = DB3.get_pars();

  % common results  
  MEX_pattern_res.x0         = x0;
  MEX_pattern_res.y0         = y0;
  MEX_pattern_res.theta0     = theta0;
  MEX_pattern_res.xM         = xM;
  MEX_pattern_res.yM         = yM;
  MEX_pattern_res.xf         = xf;
  MEX_pattern_res.yf         = yf;
  MEX_pattern_res.thetaf     = thetaf;
  MEX_pattern_res.k_max      = k_max;
  % specific results
  MEX_pattern_res.mode       = 'pattern';
  MEX_pattern_res.thetaM     = DB3_data.theta3;
  MEX_pattern_res.LA         = DB3_data.L0 + DB3_data.L1 + DB3_data.L2;
  MEX_pattern_res.LB         = DB3_data.L3 + DB3_data.L4 + DB3_data.L5;
  MEX_pattern_res.L          =  MEX_pattern_res.LA +  MEX_pattern_res.LB;
  MEX_pattern_res.type       = sign([ DB3_data.kappa0, DB3_data.kappa1, DB3_data.kappa2, DB3_data.kappa3, DB3_data.kappa4, DB3_data.kappa5 ]);
  MEX_pattern_res.typeA      = sign([ DB3_data.kappa0, DB3_data.kappa1, DB3_data.kappa2]);
  MEX_pattern_res.typeB      = sign([ DB3_data.kappa3, DB3_data.kappa4, DB3_data.kappa5]);
  MEX_pattern_res.type_word  = type2words(MEX_pattern_res.type);
  MEX_pattern_res.typeA_word = type2words(MEX_pattern_res.typeA);
  MEX_pattern_res.typeB_word = type2words(MEX_pattern_res.typeB);
  MEX_pattern_res.time       = elapsed;
  MEX_pattern_res.niter      = DB3.num_evaluation();

  % 'pattern_trichotomy'

  tic();
  flag = DB3.build( x0, y0, theta0, xM, yM, xf, yf, thetaf, k_max, 'pattern_trichotomy' );
  elapsed = toc();
  DB3_data = DB3.get_pars();

  MEX_trichotomy_res.mode = 'pattern_trichotomy';
  MEX_trichotomy_res.thetaM = DB3_data.theta3;
  MEX_trichotomy_res.LA     = DB3_data.L0 + DB3_data.L1 + DB3_data.L2;
  MEX_trichotomy_res.LB     = DB3_data.L3 + DB3_data.L4 + DB3_data.L5;
  MEX_trichotomy_res.L      = MEX_trichotomy_res.LA + MEX_trichotomy_res.LB;
  MEX_trichotomy_res.type   = sign([ DB3_data.kappa0, DB3_data.kappa1, DB3_data.kappa2, DB3_data.kappa3, DB3_data.kappa4, DB3_data.kappa5 ]);
  MEX_trichotomy_res.typeA  = sign([ DB3_data.kappa0, DB3_data.kappa1, DB3_data.kappa2]);
  MEX_trichotomy_res.typeB  = sign([ DB3_data.kappa3, DB3_data.kappa4, DB3_data.kappa5]);
  MEX_trichotomy_res.type_word  = type2words(MEX_trichotomy_res.type);
  MEX_trichotomy_res.typeA_word = type2words(MEX_trichotomy_res.typeA);
  MEX_trichotomy_res.typeB_word = type2words(MEX_trichotomy_res.typeB);
  MEX_trichotomy_res.time  = elapsed;
  MEX_trichotomy_res.niter = DB3.num_evaluation();


  %copy results
  PS_res  = MEX_pattern_res;
  PS_res.mode = 'pattern_search_MATLAB';

  PSC_res = MEX_pattern_res;
  PSC_res.mode = 'pattern_search_clustering_748_MATLAB';

  MS_res  = MEX_pattern_res;
  MS_res.mode = 'pattern_search_748_MATLAB';




  thetaGuess = (atan2(yM-y0,xM-x0) + atan2(yf-yM,xf-xM)) / 2;

  % non-derivative algorithm to find the optimal thetaM0 (residual = 0)

  L  = @(thetaM) len_Dubins(x0, y0, theta0, xM, yM, thetaM, k_max ) + ...
                len_Dubins(xM, yM, thetaM, xf, yf, thetaf, k_max );
  DL = @(thetaM) len_Dubins_DR(x0, y0, theta0, xM, yM, thetaM, k_max ) + ...
                len_Dubins_DL(xM, yM, thetaM, xf, yf, thetaf, k_max );
  typeS = @(thetaM) [ type_Dubins(x0, y0, theta0, xM, yM, thetaM, k_max ) ; ...
                      type_Dubins(xM, yM, thetaM, xf, yf, thetaf, k_max ) ];
  tic();
  [thetaM0,niter] = pattern_search(L);
  elapsed = toc();

  DB_A.build( x0, y0, theta0,  xM, yM, thetaM0, k_max );
  DB_B.build( xM, yM, thetaM0, xf, yf, thetaf,  k_max );

  PS_res.thetaM = thetaM0;
  PS_res.LA     = DB_A.length();
  PS_res.LB     = DB_B.length();
  PS_res.L      = PS_res.LA + PS_res.LB;
  PS_res.type   = typeS(thetaM0);
  [CA1,CA2,CA3] = DB_A.get_circles();
  [CB1,CB2,CB3] = DB_B.get_circles();
  TA1 = sign(CA1.kappa_begin());
  TA2 = sign(CA2.kappa_begin());
  TA3 = sign(CA3.kappa_begin());
  TB1 = sign(CB1.kappa_begin());
  TB2 = sign(CB2.kappa_begin());
  TB3 = sign(CB3.kappa_begin());
  PS_res.type  = [TA1,TA2,TA3,TB1,TB2,TB3];
  PS_res.typeA = [TA1,TA2,TA3];
  PS_res.typeB = [TB1,TB2,TB3];
  PS_res.type_word  = type2words(PS_res.type);
  PS_res.typeA_word = type2words(PS_res.typeA);
  PS_res.typeB_word = type2words(PS_res.typeB);
  PS_res.niter = niter;
  PS_res.time  = elapsed;

  %
  tic();
  [thetaM0,niter] = mixed_strategy_clust(L, DL, typeS, 1e-12, 1000);
  elapsed = toc();

  DB_A.build( x0, y0, theta0,  xM, yM, thetaM0, k_max );
  DB_B.build( xM, yM, thetaM0, xf, yf, thetaf,  k_max );

  PSC_res.thetaM = thetaM0;
  PSC_res.LA     = DB_A.length();
  PSC_res.LB     = DB_B.length();
  PSC_res.L      = PSC_res.LA + PSC_res.LB;
  PSC_res.type   = typeS(thetaM0);
  [CA1,CA2,CA3] = DB_A.get_circles();
  [CB1,CB2,CB3] = DB_B.get_circles();
  TA1 = sign(CA1.kappa_begin());
  TA2 = sign(CA2.kappa_begin());
  TA3 = sign(CA3.kappa_begin());
  TB1 = sign(CB1.kappa_begin());
  TB2 = sign(CB2.kappa_begin());
  TB3 = sign(CB3.kappa_begin());
  PSC_res.type  = [TA1,TA2,TA3,TB1,TB2,TB3];
  PSC_res.typeA = [TA1,TA2,TA3];
  PSC_res.typeB = [TB1,TB2,TB3];
  PSC_res.type_word  = type2words(PSC_res.type);
  PSC_res.typeA_word = type2words(PSC_res.typeA);
  PSC_res.typeB_word = type2words(PSC_res.typeB);
  PSC_res.niter = niter;
  PSC_res.time  = elapsed;


  %
  tic();
  [thetaM0,niter] = mixed_strategy748(L, DL, typeS, 1e-12, 1000);
  elapsed = toc();

  DB_A.build( x0, y0, theta0,  xM, yM, thetaM0, k_max );
  DB_B.build( xM, yM, thetaM0, xf, yf, thetaf,  k_max );

  MS_res.thetaM = thetaM0;
  MS_res.LA     = DB_A.length();
  MS_res.LB     = DB_B.length();
  MS_res.L      = MS_res.LA + MS_res.LB;
  MS_res.type   = typeS(thetaM0);
  [CA1,CA2,CA3] = DB_A.get_circles();
  [CB1,CB2,CB3] = DB_B.get_circles();
  TA1 = sign(CA1.kappa_begin());
  TA2 = sign(CA2.kappa_begin());
  TA3 = sign(CA3.kappa_begin());
  TB1 = sign(CB1.kappa_begin());
  TB2 = sign(CB2.kappa_begin());
  TB3 = sign(CB3.kappa_begin());
  MS_res.type  = [TA1,TA2,TA3,TB1,TB2,TB3];
  MS_res.typeA = [TA1,TA2,TA3];
  MS_res.typeB = [TB1,TB2,TB3];
  MS_res.type_word  = type2words(MS_res.type);
  MS_res.typeA_word = type2words(MS_res.typeA);
  MS_res.typeB_word = type2words(MS_res.typeB);
  MS_res.niter = niter;
  MS_res.time  = elapsed;

  results = {MEX_pattern_res, MEX_trichotomy_res, PS_res, PSC_res, MS_res};


  print_statistics(results);

  
  if flag_plot
    clf(gcf);


    subplot(2,2,1);

    hold on;
    DB_A.plot();
    DB_B.plot();
    plot([x0,xM,xf],[y0,yM,yf],'o','MarkerSize',15,'MarkerFaceColor','red');
    axis equal
    grid on

    subplot(2,2,2);

    hold on;
    DB3.plot();
    plot([x0,xM,xf],[y0,yM,yf],'o','MarkerSize',15,'MarkerFaceColor','red');
    axis equal
    grid on


    npts     = 1000;
    thetas   = linspace(thetaGuess-pi,thetaGuess+pi,npts);
    LAB      = zeros(1,npts);
    D_LAB    = zeros(1,npts);
    D_LAB_FD = zeros(1,npts);

    for i =1:npts
      th          = thetas(i);
      LAB(i)      = L(th);
      D_LAB(i)    = DL(th);
      D_LAB_FD(i) = (L(th+epsilon)-L(th-epsilon))/(2*epsilon);
    end
    IDX = find( abs(D_LAB_FD) > 5 );
    D_LAB_FD(IDX) = NaN;


    subplot(2,2,3);
    plot(thetas,LAB,'LineWidth',3);
    hold on
    plot(thetaGuess,L(thetaGuess),'o','MarkerSize',15,'MarkerFaceColor','magenta');
    plot(thetaM0,L(thetaM0),'o','MarkerSize',15,'MarkerFaceColor','blue');
    [min_tmp, min_idx] = min(LAB);
    plot(thetas(min_idx),min_tmp,'o','MarkerSize',10,'MarkerFaceColor','green');
    grid on


    subplot(2,2,4);
    plot(thetas,D_LAB,'LineWidth',3);
    hold on
    plot(thetas,D_LAB_FD,'LineWidth',2);
    plot(thetaM0,DL(thetaM0),'o','MarkerSize',10,'MarkerFaceColor','green');
    grid on
    legend({'DL','min','DL_FD'})
  end

end

%%
function print_statistics( results )
  fprintf('Results:\n');
  fprintf('  %-10s %-20s %-20s %-20s %-20s %-20s\n','VAL','MEX PS', 'MEX PS Try' ,'PS','PSC 748','PS 748');
  fprintf('  %-10s %-20s %-20s %-20s %-20s %-20s\n','---','---','---','---','---','---');
  % thetaM0
  fprintf('  %-10s %-20.15g %-20.15g %-20.15g %-20.15g %-20.15g\n','thetaM0',results{1}.thetaM,results{2}.thetaM,results{3}.thetaM,results{4}.thetaM,results{5}.thetaM);
  % total length
  fprintf('  %-10s %-20.15g %-20.15g %-20.15g %-20.15g %-20.15g\n','L', results{1}.L, results{2}.L, results{3}.L, results{4}.L, results{5}.L);
  % solution type
  fprintf('  %-10s %-20s %-20s %-20s %-20s %-20s\n','type',results{1}.type_word,results{2}.type_word,results{3}.type_word,results{4}.type_word,results{5}.type_word);
  % number of iterations
  fprintf('  %-10s %-20.15g %-20.15g %-20.15g %-20.15g %-20.15g\n','niter',results{1}.niter,results{2}.niter,results{3}.niter,results{4}.niter,results{5}.niter);
  % time
  fprintf('  %-10s %-20.15g %-20.15g %-20.15g %-20.15g %-20.15g\n','time',results{1}.time,results{2}.time,results{3}.time,results{4}.time,results{5}.time);
  fprintf('  %-10s %-20s %-20s %-20s %-20s %-20s\n','---','---','---','---','---','---');
end

%%
function len = len_Dubins( x0, y0, theta0, xf, yf, thetaf, k_max )
  DB = Dubins();
  DB.build( x0, y0, theta0, xf, yf, thetaf, k_max );
  [len,~,~] = DB.length();
end

%%
function DL = len_Dubins_DL( x0, y0, theta0, xf, yf, thetaf, k_max )
  DB = Dubins();
  DB.build( x0, y0, theta0, xf, yf, thetaf, k_max );
  [~,DL,~] = DB.length();
end

%%
function DR = len_Dubins_DR( x0, y0, theta0, xf, yf, thetaf, k_max )
  DB = Dubins();
  DB.build( x0, y0, theta0, xf, yf, thetaf, k_max );
  [~,~,DR] = DB.length();
end

%%
function type = type_Dubins( x0, y0, theta0, xf, yf, thetaf, k_max )
  DB = Dubins();
  DB.build( x0, y0, theta0, xf, yf, thetaf, k_max );
  type = DB.curve_type();
end

%% function to perform a pattern search to find the optimal thetaM0
function [thetaM0, n_iter] = pattern_search( func )
  % Initialize variables
  theta_max       = +pi;
  theta_min       = -pi;
  theta_candidate = 0;
  numpts          = 16;
  delta_theta     = (theta_max - theta_min) / numpts;
  min_residual    = Inf;
  n_iter          = 0;
  while delta_theta > 1e-16
    thetas = linspace(theta_min, theta_max, numpts);
    residuals = zeros(1, numpts);
    for i = 1:length(thetas)
      thetaM0      = thetas(i);
      residuals(i) = func(thetaM0);
      n_iter       = n_iter + 1;
    end
    [min_residual, min_idx] = min(residuals);
    theta_candidate = thetas(min_idx);
    theta_min       = theta_candidate-delta_theta;
    theta_max       = theta_candidate+delta_theta;
    delta_theta     = (theta_max - theta_min) / numpts;
  end
  thetaM0 = theta_candidate;
end

%% function to compute the solution with pattern search + Algo748
function [root, tot_iter] = mixed_strategy748(func, d_func, ftype, tol, max_iter)
  % Initialize variables
  theta_max       = pi;
  theta_min       = -pi;
  theta_candidate = 0;
  numpts          = 16;
  delta_theta     = (theta_max - theta_min) / numpts;
  iter = 0;
  subiter = 0;
  A748 = Algo748();
  while (delta_theta > tol && iter < max_iter)
    thetas = linspace(theta_min, theta_max, numpts);
    residuals = zeros(1, numpts);
    residuals = arrayfun(func, thetas);
    [~, min_idx] = min(residuals);
    theta_candidate = thetas(min_idx);
    theta_min       = theta_candidate-delta_theta;
    theta_max       = theta_candidate+delta_theta;
    delta_theta     = (theta_max - theta_min) / numpts;
    iter = iter + numpts;
    numpts = max(4,numpts/2);
    type_min   = ftype(theta_min);
    type_max   = ftype(theta_max);
    d_func_min = d_func(theta_min);
    d_func_max = d_func(theta_max);
    if (type_min(1) == type_max(1) && type_min(2) == type_max(2) && sign(d_func_min) ~= sign(d_func_max))
      [root, subiter] = A748.eval(theta_min, theta_max, d_func);
      tot_iter = iter + subiter;
      %fprintf("Mixed Strategy (pattern + Algo748): Solution found at iter %d\n",tot_iter);
      return;
    end
  end
  root = theta_candidate;
  tot_iter = iter + subiter;
  %fprintf("Mixed Strategy (pattern): Solution found at iter %d\n",tot_iter);
end


%% pattern search clustering plus Algo748

function [root, tot_iter] = mixed_strategy_clust(func, d_func, ftype, tol, max_iter)
  % Initialize variables
  theta_max   = +pi + 2*pi/16;
  theta_min   = -pi - 2*pi/16;
  numpts      = 18; %16
  thetas      = linspace(theta_min, theta_max, numpts);
  Ls          = arrayfun(func, thetas);
  DLs         = arrayfun(d_func, thetas);
  Types       = arrayfun(ftype, thetas, 'UniformOutput', false);
  iter        = numpts;
  clusters    = {};
  [F_theta_candidate, min_idx] = min(Ls);
  theta_candidate = thetas(min_idx);
  % check the pattern high low high of the Ls
  for i = 2:length(Ls)-1
    im = i-1;
    ip = i+1;
    if (Ls(im) >= Ls(i) && Ls(i) <= Ls(ip)) 
      %fprintf("Cerca a i = %d\n",i);
      tmp_struct.theta = [thetas(im), thetas(i), thetas(ip)];
      tmp_struct.L     = [Ls(im), Ls(i), Ls(ip)];
      tmp_struct.dL    = [DLs(im), DLs(i), DLs(ip)];
      tmp_struct.type  = [Types(im), Types(i), Types(ip)];
      clusters{end+1} = tmp_struct;
    end
  end
  %
  tot_iter = iter;
  %
  F_root = F_theta_candidate;
  queue = clusters; % Initialize the queue with the initial clusters
  while ~isempty(queue)
    cluster = queue{1};
    queue(1) = []; % Dequeue the first cluster

    if cluster.theta(1) > pi && cluster.theta(2) > pi && cluster.theta(3) > pi 
      cluster.theta = cluster.theta-2*pi;
    end
    if cluster.theta(1) < -pi && cluster.theta(2) < -pi && cluster.theta(3) < -pi 
      cluster.theta = cluster.theta+2*pi;
    end
    % Check if the cluster has the same type of solution and the derivative at the border have opposite signs
    if cluster.type{1}(1) == cluster.type{2}(1) && cluster.type{2}(1) == cluster.type{3}(1) && ...
       cluster.type{1}(2) == cluster.type{2}(2) && cluster.type{2}(2) == cluster.type{3}(2) && ...
       (cluster.dL(1)) * (cluster.dL(3)) <= 0
      A748 = Algo748();
      [root, subiter] = A748.eval(cluster.theta(1), cluster.theta(3), d_func);
      if root > pi  
        root = root-2*pi;
      elseif root < -pi 
        root = root+2*pi;
      end
      tot_iter = tot_iter + subiter;
      F_root = func(root);
      if F_root < F_theta_candidate
        theta_candidate = root;
        F_theta_candidate = F_root;
      end
      continue;
    end
    % From cluster compute additional points between theta(1) and theta(3)
    thetas = [cluster.theta(1), (cluster.theta(1) + cluster.theta(2)) / 2, cluster.theta(2), (cluster.theta(2) + cluster.theta(3)) / 2, cluster.theta(3)];
    Ls     = [cluster.L(1), func(thetas(2)), cluster.L(2), func(thetas(4)), cluster.L(3)];
    DLs    = [cluster.dL(1), d_func(thetas(2)), cluster.dL(2), d_func(thetas(4)), cluster.dL(3)];
    Types  = {cluster.type{1}, ftype(thetas(2)), cluster.type{2}, ftype(thetas(4)), cluster.type{3}};
    iter = 2;
    new_clusters = {};
    for i = 2:length(Ls)-1
      if Ls(i-1) >= Ls(i) && Ls(i) <= Ls(i+1)
        tmp_struct.theta = [thetas(i-1), thetas(i), thetas(i+1)];
        tmp_struct.L     = [Ls(i-1), Ls(i), Ls(i+1)];
        tmp_struct.dL    = [DLs(i-1), DLs(i), DLs(i+1)];
        tmp_struct.type  = [Types(i-1), Types(i), Types(i+1)];
        new_clusters{end+1} = tmp_struct;
      end
    end
    tot_iter = tot_iter + iter;
    for i = 1:length(new_clusters)
      i_cluster = new_clusters{i};
      if (abs(i_cluster.theta(1) - i_cluster.theta(2)) < tol || abs(i_cluster.theta(2) - i_cluster.theta(3)) < tol) || tot_iter > max_iter
        F_root = i_cluster.L(2);
        root = i_cluster.theta(2);
        subiter = 0;
        if F_root < F_theta_candidate
          theta_candidate = root;
          F_theta_candidate = F_root;
        end
      else
        queue{end+1} = i_cluster; % Enqueue the new cluster
        subiter = 0; % No iteration in this loop step
      end
      tot_iter = tot_iter + subiter;
    end
  end
  root = theta_candidate;
end


%% function to compute the type of the solution
function out = type2words( type )
  out = '';
  for k=1:length(type)
    if type(k) == 0
      out = strcat(out,'S');
    elseif type(k) == 1
      out = strcat(out,'L');
    else
      out = strcat(out,'R');
    end
  end
end

%% read txt file
function dataTable = read_SACON_file()
  % % read txt file  
  filename = './saccon_data/test3P_360_0.txt';

  fid = fopen(filename, 'r');

  % Read the first line to get the headers
  headers = fgetl(fid);
  headers = strsplit(strtrim(headers)); % Remove leading/trailing whitespace and split into headers

  headers = { 'Pi_x', 'Pi_y', 'Pi_th', 'Pm_x', 'Pm_y', 'Pm_th', 'Pf_x', 'Pf_y', 'Pf_th', 'kmax', 'Length', 'Time', 'd1_words', 'd2_words', 'd1_s1', 'd1_s2', 'd1_s3', 'd2_s1', 'd2_s2', 'd2_s3' };

  stringColumns = find(contains(headers, 'words'));

  % Initialize a cell array to store the even lines
  data = [];

  % Start reading from the second line
  lineNumber = 2;
  while ~feof(fid)
      line = fgetl(fid);
      if mod(lineNumber, 2) == 0 % Check if the line number is even
          lineData = strsplit(strtrim(line)); % Remove leading/trailing whitespace and split into data fields
          if length(lineData) < 10
            continue;
          end
          % Convert each field to the appropriate type
          for i = 1:length(lineData)
              if ~ismember(i, stringColumns)
                  lineData{i} = str2double(lineData{i}); % Convert to double if not a string column
              end
          end
          data = [data; lineData]; % Append the converted data to the cell array
      end
      lineNumber = lineNumber + 1;
  end

  % Close the file
  fclose(fid);

  % Convert cell array to table
  dataTable = cell2table(data, 'VariableNames', headers);

end

%% function to print SACCON data

function print_saccon_data(dataTabl,i)
  fprintf('SACCON data %d/%d\n',i,size(dataTabl,1));
  fprintf('Pi = (%g,%g,%g)\n',dataTabl.Pi_x(i),dataTabl.Pi_y(i),dataTabl.Pi_th(i));
  fprintf('Pm = (%g,%g,%g)\n',dataTabl.Pm_x(i),dataTabl.Pm_y(i),dataTabl.Pm_th(i));
  fprintf('Pf = (%g,%g,%g)\n',dataTabl.Pf_x(i),dataTabl.Pf_y(i),dataTabl.Pf_th(i));
  fprintf('kmax = %g\n',dataTabl.kmax(i));
  % thetaM0
  fprintf('  %-10s %-20.15g\n','thetaM0',dataTabl.Pm_th(i));
  % total length
  fprintf('  %-10s %-20.15g\n','L', dataTabl.Length(i));
  % solution type
  fprintf('  %-10s %-20s\n','d1',[dataTabl.d1_words{i},dataTabl.d2_words{i}]);
  % number of iterations
  fprintf('  %-10s %-20s\n','niter','???');
  % time
  fprintf('  %-10s %-20.15g\n','time',dataTabl.Time(i));


end