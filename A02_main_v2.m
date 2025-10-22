idxs = contains(ema_all.Properties.VariableNames, '_x') |...
       contains(ema_all.Properties.VariableNames, '_y') ; 
X = table2array(ema_all(:, idxs));

numSensors = 6;  % Total sensors inferred from 12 channels (each sensor: x and y)
% Reshape the mean vector into (numSensors x 2) where column 1 is x and column 2 is y.
% sensorNeutral = reshape(mu, 2, numSensors)';  

%% --- Normalization ---
% % Subtract the mean from each channel to obtain zero-mean (centered) data.
% % This form of normalization is common prior to PCA.
% mu = mean(X, 1);             % 1x12 vector: mean for each channel across all samples
% X_centered = X - mu;         % Center the data


% for data that are already normalized
% mu = mean(X, 1);             % 1x12 vector: mean for each channel across all samples
X_centered = X;      

%% --- Principal Component Analysis (PCA) ---
% Perform PCA on the centered data.
% coeff: principal component coefficients (loadings) (12x12 matrix)
% score: the representation of X_centered in the PC space.
% latent: variance explained (eigenvalues)
% explained: percentage of total variance explained by each PC.
[coeff, score, latent, ~, explained] = pca(X_centered);

%% --- Plot Cumulative Variance Explained ---
cumVar = cumsum(explained);
figure;
plot(cumVar, 'o-', 'LineWidth', 2);
xlabel('Number of Components');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance vs. Number of Components');
grid on;

%% --- Visualization: Principal Component Directions ---
% For visualization purposes, we return to the neutral sensor positions (the mean)
% and add the PC directions (from the loading vectors). 
%
% Assumption: The 12 columns in X correspond to 6 sensors. Each sensor is measured in [x, y]
% in sequential columns. So we reshape the mean vector and PC coefficients accordingly.


% We will visualize the directions for the top 6 PCs.
figure; tiledlayout(3,3, 'TileSpacing','tight'); 
for p = 1:9
    % subplot(2, 5, p);
    nexttile(p);
    
    % Reshape the current PC coefficient vector into a (numSensors x 2) matrix.
    % Each row now corresponds to a sensor's (x, y) "movement direction" indicated by this PC.
    pcCoeffs = reshape(coeff(:,p), 2, numSensors)';  
    pcCoeffs = pcCoeffs*2.5; % scale for visualization
    
    % Create a quiver plot: draw arrows starting at the neutral sensor positions 
    % with directions given by pcCoeffs.
    quiver(sensorNeutral(:,1), sensorNeutral(:,2), pcCoeffs(:,1), pcCoeffs(:,2), 0, 'LineWidth', 2);
    hold on;
    
    % Plot the neutral positions (as circles) for clarity.
    scatter(sensorNeutral(:,1), sensorNeutral(:,2), 50, 'filled');
    title(['Principal Component ' num2str(p)]);
    xlabel('X Position');
    ylabel('Y Position');
    axis equal;
    grid on;
    hold off;
end


%%  Create animation
ema_subset = ema_all(ema_all.file_id==10, :); 

X = table2array(ema_subset(:, 3:end)) + sensorNeutralVector; 
time_X = table2array(ema_subset(:, 1));

% downsample
X = X(1:4:end, :); 
time_X = time_X(1:4:end, :); 

X_PC = X * PC; X_recon = X_PC * PCinv; 
% X = X_recon; 

% % subet for visualization
% idxs = 20e3 + (1:10e2);
% X = X(idxs, :); 
% time_X = time_X(idxs, :); 

% Parameters for the trail effect
trailLength = 5; % Number of frames in the trail
numFrames = size(X, 1);

% Prepare video writer object
gifFile = 'myAnimation.gif';

outputFilename = './fig/ema_sensor_motion.avi';
v = VideoWriter(outputFilename);
v.FrameRate = FS_EMA; % Adjust as needed
open(v);

% Setup figure
figure('Position',[100 100 600 600]);
hold on;
axis equal;
grid on;
xlabel('X Position');
ylabel('Y Position');
title('EMA Sensor Motion with Trails');

% Determine plot limits
padding = 1;
xlim([min(X(:,1:2:end),[],'all')-padding, max(X(:,1:2:end),[],'all')+padding]);
ylim([min(X(:,2:2:end),[],'all')-padding, max(X(:,2:2:end),[],'all')+padding]);

exportgraphics(gcf, gifFile);

% Initialize data structure to store previous positions
sensorHistory = nan(numSensors, 2, trailLength);

% Main loop to generate frames
for frameIdx = 1:numFrames
    cla; % Clear current frame
    
    % Get current sensor positions
    currentFrame = reshape(X(frameIdx,:), 2, [])';

    % Update sensor history (for trail effect)
    sensorHistory(:,:,2:end) = sensorHistory(:,:,1:end-1);
    sensorHistory(:,:,1) = currentFrame;
    
    % Plot trails for each sensor
    for sIdx = 1:numSensors
        % Extract history for the current sensor
        trailX = squeeze(sensorHistory(sIdx,1,:));
        trailY = squeeze(sensorHistory(sIdx,2,:));
        
        % Plot trail (comet-like effect)
        validIdx = ~isnan(trailX);
        plot(trailX(validIdx), trailY(validIdx), '-o', 'LineWidth',2,...
            'MarkerSize',8,'MarkerFaceColor',[0 0.4470 0.7410]);
        
        % Label current sensor position (only at latest position)
        if validIdx(1)
            text(trailX(1)+0.3, trailY(1)+0.3, sensorLabels{sIdx},...
                'FontSize',10,'FontWeight','bold','Color','k');
        end
    end

    text('Units','normalized','Position',[0.5 0.9], 'FontSize', 16, ...
        'String', sprintf('time: %0.2f', time_X(frameIdx))) 

    % Capture and write frame to video
    frame = getframe(gcf);
    % writeVideo(v, frame);

    exportgraphics(gcf, gifFile, Append=true);
end

% Close video writer
% close(v);
close(gcf);

disp(['Animation saved successfully to: ' outputFilename]);

%% --- Optional: Reconstructing the Data with PC Dimensions ---
% As an example, you can reconstruct the data by adding some variation along one of the PCs.
% For instance, vary the first PC by a scaling factor and add it back to the neutral sensor positions.
pcIndex = 1;
scalingFactor = 5;  % Adjust to see more or less movement
% The perturbation for sensor positions: reshape the PC coefficients for pcIndex.
pcPerturbation = reshape(coeff(:,pcIndex), 2, numSensors)' * scalingFactor;
% New sensor positions with perturbation along the first PC.
sensorPerturbed = sensorNeutral + pcPerturbation;

figure;
% Plot neutral sensor positions.
plot(sensorNeutral(:,1), sensorNeutral(:,2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
hold on;
% Plot perturbed sensor positions.
plot(sensorPerturbed(:,1), sensorPerturbed(:,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
% Use quiver to show the displacement.
for i = 1:numSensors
    quiver(sensorNeutral(i,1), sensorNeutral(i,2), pcPerturbation(i,1), pcPerturbation(i,2), 0, 'r', 'LineWidth', 2);
end
title('Neutral (Black) and Perturbed (Red) Sensor Positions (PC 1 Scaling)');
xlabel('X Position');
ylabel('Y Position');
legend('Neutral', 'Perturbed');
axis equal;
grid on;
hold off;