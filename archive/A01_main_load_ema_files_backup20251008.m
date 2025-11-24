% You are an expert in cognitive neuroscience, articulatory phonetics, machine learning, 
% and control theory. I'm a student work on a pilot project concerned with representation
% learning with  electromagntic articulography data.
% 
% One problem with electromagnetic articulography (EMA) is that the EMA x, y postitions 
% don't represent coherent vocal tract movements. For example, if we 'perturb' one dimensions 
% of an EMA sensor (say, LL_x), there must necessarily be consequences for the other 
% articualtors. We are trying to find a subspace that represents coherent dimensions that respect
% the anatomy of the vocal tract. 
% 
% The goal of this pilot experiment is to see if we can find a representation 
% of EMA sensor data that can be used 
% to act as a 'controller' to represent and perturb vocal tract data. 
% 
% I have an EMA dataset from the mngu0 project. this is the csv we load. 
% Each articulator is coded as <ART>_<x/y>, where ART is the articulator
% name, and then there are two columns defining the x and y positions for
% that articulator. T3 is tongue dorsum, T2 is tongue body, and T1 is
% tongue tip. UL is upper lip. LL is lower lip. 
% Every row represents a sample from the EMA system.
% 
% The representatino we are seeking is likely related
% to the concept of 'constrictions' in vocal tract anatomy and articulatory phonetics. 
% For example, we might want to define a 'lip aperture' dimension
%  that moves all of the EMA x,y sensor positions coherently to create a lip constriction. 
%  Another example might be a 'velar constriction' dimension. This dimension, when scaled, 
%  should move the EMA x,y postitions coherently to create a constriction at the velum. 
%  This velar constriction would have large coefficients for the T3 'tongue dorsum'
%  EMA sensor and probably lower coefficients for EMA sensors that are less concerned
%  with velar constriction like the lower lip 'LL' sensor.
% 
%  JAW_x  and JAW_y will largely determine the positions of the other articulators (besides the upper lip UL articulators)
%  because they are physically connected to the Jaw . 
% One idea is to represent All of the other positions besides UL  with respect to this 
% JAW axis (via change of basis?). This would be similar to how a robot controller might represent multiple joints, 
%  where a secondary distal joint would be defined in terms of the angle of a primary distal
%  joint. Another area of study to draw on would be pose estimation and human motion
%  estimation. 
%  If you have any ideas as to how to represent these EMA data, I'm open to further 
%  discussion. 
% 
% 
%  Let's define these dimensions as 'principle components of motion' PCMs. Every speech 
%  sound (for example, a /p/) can be represented as a time series of the PCMs.
% 
%  I'm not sure I've through all of the possibilities here, so feel free to take liberties
%  in designing the system. The important thing is that we achieve the goal; the steps along the
%  way are less concrete. I'm just laying out some ideas. 
% 
%  Other considerations: 
% 
% How can we account for sensors having 2 covariates each, which are correlated? 
% 
% 
% We might consider modeling the derivative (deltas, representing velocity) 
% rather than the absolute position of each articulator. 
% 
% 
% Can you first generate ideas as to how to proceed? We can settle on a plan together.
% Then, I will ask for help in writing code to achieve this. 
% 

%% Set up and load EMA data

% for normalized data
CHANNEL_NAMES = readtable('channel_names_norm.csv', 'Delimiter', ',');
sensorLabels = {'T3', 'T2', 'T1', 'JAW', 'UL', 'LL'};
chans_of_interest = [string(sensorLabels)' + "_x" string(sensorLabels)' + "_y"]';
chans_of_interest = chans_of_interest(:);
chans_of_interest_idxs = ismember(CHANNEL_NAMES{:, 2}, chans_of_interest);
CHANNEL_NAMES(chans_of_interest_idxs,:)


% % for basic data
% CHANNEL_NAMES = readtable('channel_names_basic.csv', 'Delimiter', ',');
% sensorLabels = {'T3', 'T2', 'T1', 'jaw', 'upperlip', 'lowerlip'};
% chans_of_interest = [string(sensorLabels)' + "_px" string(sensorLabels)' + "_py"]';
% chans_of_interest = chans_of_interest(:);
% chans_of_interest_idxs = ismember(CHANNEL_NAMES{:, 2}, chans_of_interest);
% CHANNEL_NAMES(chans_of_interest_idxs,:)


% chans_of_interest_idxs = contains(ema_all.Properties.VariableNames, '_x_d') |...
%        contains(ema_all.Properties.VariableNames, '_y') ; 
% chans_of_interest = CHANNEL_NAMES(chans_of_interest_idxs, :);

FS_EMA = 200; % Hz
path_data = '/Users/ly546/Documents/data/'; 
data0_type = 'norm'; % 'norm' 'basic'
if data0_type=="norm"; files = struct2table(dir([path_data filesep 'mngu0_s1_ema_norm_1.0.1'])); % normalized
elseif  data0_type=="basic"; files = struct2table(dir([path_data filesep 'mngu0_s1_ema_basic_1.1.0'])); % unnormalized EMA positions
end

files = files(contains(files.name, '.ema'),  :);

ema_all = []; 
meta_all = [];
for ifile = 1:height(files)
    ema_current = estload([files.folder{ifile} filesep files.name{ifile}]);
    assert(width(ema_current==36)); 
    
    meta_current = repmat(ifile, [height(ema_current), 1]); 
    time = (1:height(ema_current)) / FS_EMA;
    meta_current = [time' meta_current]; 

    if isempty(ema_all); 
        ema_all = ema_current;
        meta_all = meta_current;
    else 
        ema_all = [ema_all; ema_current]; 
        meta_all = [meta_all; meta_current]; 
    end
    
    % testing
    if ifile > 500; break; end
end


ema_all = ema_all(:, chans_of_interest_idxs);
ema_all = array2table([meta_all ema_all], 'VariableNames', ['time', 'file_id', convertStringsToChars(chans_of_interest)' ]); 


writetable(ema_all, [path_data filesep 'test_ema_' data0_type '.csv']); 


%% Read and Set Sensor Neutral Positions from file


if data0_type=="norm";
    % for normed data
    % Read numerical data from the file ema_means.txt
    % filePath = './data/norm_params/ema_means.txt';
    % filaePath = [path_data filesep 'mngu0_s1_ema_norm_1.0.1/norm_params/ema_means.txt']
    filePath = '/Users/ly546/Documents/data/mngu0_s1_ema_norm_1.0.1/norm_parms/ema_means.txt';
    fid = fopen(filePath, 'r');
    emaMeans = fscanf(fid, '%f')';
    fclose(fid);
    % Take the first 12 values (assuming the first 12 correspond to the EMA sensor positions)
    sensorNeutralVector = emaMeans(1:12);

elseif  data0_type=="basic";
    % for basic data
    idxs = contains(ema_all.Properties.VariableNames, '_px') |...
        contains(ema_all.Properties.VariableNames, '_py') ;
    X = table2array(ema_all(:, idxs));
    sensorNeutralVector = mean(X, 1, 'omitmissing');

end




% Reshape into 6 sensors x 2 dimensions (x,y)
sensorNeutral = reshape(sensorNeutralVector, 2, [])';

% Plot neutral sensor positions with labels
figure;
scatter(sensorNeutral(:,1), sensorNeutral(:,2), 100, 'filled', 'b');
hold on;

labelOffset = 0.2; % Adjust as needed
for i = 1:length(sensorLabels)
    text(sensorNeutral(i,1)+labelOffset, sensorNeutral(i,2)+labelOffset, ...
         sensorLabels{i}, 'FontSize', 12, 'FontWeight', 'bold');
end

title('Neutral Positions of EMA Sensors (Loaded from File)');
xlabel('X Position');
ylabel('Y Position');
axis equal;
grid on;
hold off;

%% Scatter plot of ~5 EMA frames with each articulator position labeled.
% Save each file in ./fig/example_frame_frameidx-<frame-index>.png


%% Plot movie of articulators moving over the course of frames. Take a
% random ~1000 contiguous samples to show how articulators are moving
% Save as ./fig/example_motion-<first-frame-index>.avi. 

%% Approach 1: naive approoach, don't consider JAW normalization. Simple take the data
% Compute PCA representions
% Plot the cumulative variance vs number of components. 

% For the top 6 principle components...
%     Plot the sensors in their average positions and quivers (x, y and magnitude) denoting the direction of the principle component. 


%% Run PCA on the JAW_x JAW_y. This will determine the first principle compoent of 
% motion for the system.
% Plot a figure showing the JAW PC with quivers. 

% Represent the other articulators (besdies UL) in terms of the JAW PC. 

%% Run PCA on transformed ata
% 1. Define PCs for the other components in the JAW PC space.



%% Generate simulated data. Start from the neutral position of the vocal tract and 
% then, for each PCM, perturb along the dimension and then return to the neutral position. 





