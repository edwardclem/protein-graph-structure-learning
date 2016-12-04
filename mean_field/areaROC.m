
function [aROC, alarmRate, detectRate] = areaROC(confidence, testClass, plotStyle)
%areaROC     Compute and plot ROC curve and corresponding area
%
%    [aROC, alarmRate, detectRate] = areaROC(confidence, testClass, plotStyle)
%
%  Compute ROC curve, where
%    confidence = vector of scores for each test example, where higher scores
%                 indicate greater confidence in target presence
%    testClass  = vector giving ground truth for each test example, where
%                 1 indicates target presence and 0 target absence
%    plotStyle  = text string giving style for plotted ROC curve, or
%                 empty to not create plot (DEFAULT)
%    aROC       = area under ROC curve
%    alarmRate  = false alarm rates, uniformly sampled on [0,1]
%    detectRate = corresponding detection rates computed from given confidence

if nargin < 2
  error('Insufficient number of input arguments');
end
if nargin < 3
  plotStyle = '';
end
numpoints = 500;    % number of false alarm rates at which to evaluate curve

% perturb confidence to avoid "staircase" effect
S = rand('state');
rand('state',0);
confidence = confidence + rand(size(confidence))*10^(-10);
rand('state',S)

% indices of negative and positive test cases
ndxAbs   = find(testClass <= 0); % absent
ndxPres  = find(testClass >= 1); % present
confAbs  = sort(confidence(ndxAbs));
confPres = sort(confidence(ndxPres));

% compute ROC curve
confAbsResamp = confAbs(fix(linspace(1, length(confAbs), numpoints)));

alarmRate  = zeros(1,numpoints);
detectRate = zeros(1,numpoints);
for ii = 1:numpoints
  detectRate(ii) = sum(confPres>= confAbsResamp(ii)) / length(ndxPres);
  alarmRate(ii)  = sum(confAbs >= confAbsResamp(ii)) / length(ndxAbs);
end

% compute area under ROC curve
aROC = abs(sum((alarmRate(2:end)-alarmRate(1:end-1)) .* ...
  (detectRate(2:end)+detectRate(1:end-1))/2));

% if desired, plot ROC curve
if ~isempty(plotStyle)
  plot(alarmRate, detectRate, plotStyle); 
  axis([0 1 0 1]);
  grid on;
  ylabel('Detection Rate');
  xlabel('False Alarm Rate');
end

