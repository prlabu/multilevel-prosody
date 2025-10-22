%% --- Load data ---
% data = readtable('test_ema_basic.csv');
X = table2array(ema_all);  % n_samples x 12 (6 articulators x [x,y])

%% --- Step 1: PCA on Jaw positions ---
% Extract JAW x,y (columns 11,12)
jaw_idxs = contains(ema_all.Properties.VariableNames, 'JAW'); 
jawXY = X(:, jaw_idxs);
jawMean = mean(jawXY, 1);
jawXY_centered = jawXY - jawMean;

% PCA on jaw positions
[jawCoeff, jawScore, ~, ~, jawExplained] = pca(jawXY_centered);

% Jaw Opening variable (first Jaw PC projection)
jawOpening = jawScore(:,1); % (n_samples x 1)

%% --- Step 2: Regress out Jaw Opening from all positions ---
X_residual = zeros(size(X)); % Initialize residual matrix

for i = 1:size(X,2)
    mdl = fitlm(jawOpening, X(:,i)); % Linear regression on jawOpening
    X_residual(:,i) = mdl.Residuals.Raw; % Extract residuals
end

%% --- Step 3: Split into Lip and Tongue residual matrices ---
% Lips: UL (cols 7,8), LL (cols 9,10)
lip_idxs = contains(ema_all.Properties.VariableNames, 'UL') | contains(ema_all.Properties.VariableNames, 'LL'); 
lipResiduals = X_residual(:, lip_idxs);

% Tongue: T3 (1,2), T2 (3,4), T1 (5,6)
tongue_idxs = contains(ema_all.Properties.VariableNames, 'T1') | ...
              contains(ema_all.Properties.VariableNames, 'T2') | ...
              contains(ema_all.Properties.VariableNames, 'T3');
tongueResiduals = X_residual(:, tongue_idxs);

%% --- Step 4: PCA separately on lips and tongue residuals ---
% PCA on lip residuals
[lipCoeff,~,~,~,lipExplained] = pca(lipResiduals);

% PCA on tongue residuals
[tongueCoeff,~,~,~,tongueExplained] = pca(tongueResiduals);

%% --- Plot Cumulative Variance Explained ---
jawExplained = cumsum(jawExplained);
lipExplained = cumsum(lipExplained);
tongueExplained = cumsum(tongueExplained);

figure;
plot(jawExplained, 'o-', 'LineWidth', 2, 'DisplayName', 'jaw'); hold on; 
plot(lipExplained, 'o-', 'LineWidth', 2, 'DisplayName', 'lips'); hold on; 
plot(tongueExplained, 'o-', 'LineWidth', 2, 'DisplayName', 'tongue');
xlabel('Number of Components');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance vs. Number of Components');
grid on;
legend; 



%% --- Visualization of Jaw Opening, Lip PCs, and Tongue PCs ---
% sensorLabels = {'T3', 'T2', 'T1', 'JAW', 'UL', 'LL'};

% Number of PCs to visualize
numPCs = 4;

figure;
for pcIdx = 1:numPCs
    subplot(1,numPCs,pcIdx);
    hold on;
    
    % Plot sensor neutral positions
    scatter(sensorNeutral(:,1), sensorNeutral(:,2), 50, 'filled','k');
    
    % --- Jaw Opening Quiver ---
    jaw_direction = jawCoeff(:,1)'; % (1x2) vector (direction of first jaw PC)
    jawNeutralPos = sensorNeutral(ismember(sensorLabels, {'JAW'}),:); % last row is JAW
    quiver(jawNeutralPos(1), jawNeutralPos(2),...
           jaw_direction(1), jaw_direction(2),...
           0, 'Color','g','LineWidth',2,'MaxHeadSize',1.5);
    
    % --- Lip Quivers (Lip PC directions) ---

    lipNeutral = sensorNeutral(ismember(sensorLabels, {'UL', 'LL'}),:); % UL and LL
    lipDirs = reshape(lipCoeff(:,pcIdx),2,[])'; % 2 sensors x [x,y]
    quiver(lipNeutral(:,1), lipNeutral(:,2),...
           lipDirs(:,1), lipDirs(:,2),...
           0, 'Color','b','LineWidth',2,'MaxHeadSize',1.5);
       
    % --- Tongue Quivers (Tongue PC directions) ---
    tongueNeutral = sensorNeutral(ismember(sensorLabels, {'T3', 'T2', 'T1'}),:); % T3,T2,T1
    tongueDirs = reshape(tongueCoeff(:,pcIdx),2,[])'; % 3 sensors x [x,y]
    quiver(tongueNeutral(:,1), tongueNeutral(:,2),...
           tongueDirs(:,1), tongueDirs(:,2),...
           0, 'Color','r','LineWidth',2,'MaxHeadSize',1.5);
       
    % Labels and formatting
    text(sensorNeutral(:,1)+0.3, sensorNeutral(:,2)+0.3, sensorLabels,...
         'FontSize',10,'FontWeight','bold');

    title(['PC #' num2str(pcIdx)]);
    xlabel('X Position');
    ylabel('Y Position');
    % axis equal;
    grid on;
    hold off;
    
    if pcIdx==1
        legend({'Neutral','Jaw Opening','Lip PC','Tongue PC'},'Location','best');
    end
end
sgtitle('Jaw Opening, Lip PCs, and Tongue PCs');

set(findall(gcf, 'type', 'axes'), 'xlim', [-2, 6], 'ylim', [-4, 1])

%% define the control matrix that maps PC space to EMA space
PC = zeros([12, 7]); 
PC(1:6, 1:3) = tongueCoeff(:, 1:3); 
PC(:, 4) = repmat(jawCoeff(:,1), [6 1]); % jaw opening gets added to all variables 
PC(9:12, 5:7) = lipCoeff(:, 1:3); 
PC(9:10, 4) = 0; % upperlip doesn't get controlled by jaw 
PCinv = pinv(PC); 

figure; imagesc(PC); colorbar; axis equal

%% Alternatively, define an invertible 12x12 matrix
% define the control matrix that maps PC space to EMA space
PC = zeros([12, 12]); 
PC(1:6, 1:6) = tongueCoeff; 
% PC(7:8, 7:8) = repmat(jawCoeff, [6 1]); % jaw opening gets added to all variables 
PC(7:8, 7:8) =  jawCoeff; % jaw opening gets added to all variables 
PC(9:12, 9:12) = lipCoeff; 
% PC(9:10, 7:8) = 0; % upperlip doesn't get controlled by jaw 
PC

PCinv = pinv(PC); 

figure; imagesc(PC); colorbar; axis equal

p2P(p)

idxs_mask_jawUL = ones([height(p) 1]); 
idxs_mask_jawUL(9:10) = 0; % don't want to move the upper lip
idxs_mask_jawUL(7:8) = 0; % we don't want to move the jaw either
p_jawrel = p - idxs_mask_jawUL .* repmat(p(idxs_jaw), [6, 1]); 

P = PC * p_jawrel; 

aa = array2table(PC, 'VariableNames', ema_all.Properties.VariableNames(tongue_idxs | jaw_idxs | lip_idxs), ...
    'RowNames', ema_all.Properties.VariableNames(tongue_idxs | jaw_idxs | lip_idxs))
writetable(aa, 'PC-12d.csv')


%% Convert an EMA position into PC space 


jawPos = jawPos; % 2 x 1
lipPosRel = lipPos; % 4 x 1 
lipPosRel(3:4, :) = lipPosRel(3:4, :) - jawPos;  % 2 x 2
tonguePosRel = tonguePos - repmat(jawPos, [3 1]); % 6  x 1

jawP = jawCoeff' * jawPos;
lipP = lipCoeff' * lipPosRel;
tongueP = tongueCoeff' * tonguePosRel;



%%
interactiveEMA_GUI_matrix(sensorNeutral, sensorLabels, PC);

function interactiveEMA_GUI_matrix(sensorNeutral, sensorLabels, PC)
    % GUI for EMA with Jaw-dependent articulator positions

    % Create figure
    fig = figure('Name','Interactive Vocal Tract Control',...
                 'NumberTitle','off','Position',[100 100 1200 600]);

    % Slider initial values
    slider_labels = {'Jaw Opening','Lip PC1','Lip PC2','Lip PC3',...
                     'Tongue PC1','Tongue PC2','Tongue PC3'};
    sliders = gobjects(1,7);
    slider_range = [-3 3];

    % Create sliders and labels
    for i=1:7
        sliders(i) = uicontrol('Parent',fig,'Style','slider',...
                     'Min',slider_range(1),'Max',slider_range(2),'Value',0,...
                     'Units','normalized','Position',[0.05, 0.9-0.1*i, 0.3, 0.04],...
                     'Callback', @updatePositions);

        uicontrol('Parent',fig,'Style','text','Units','normalized',...
                  'Position',[0.05, 0.94-0.1*i, 0.3, 0.03],'String',slider_labels{i},...
                  'FontSize',10,'HorizontalAlignment','left');
    end

    % Axes setup
    ax = axes('Parent',fig,'Position',[0.4 0.1 0.55 0.8]);
    hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
    xlim(ax,[-2, 6]); ylim(ax,[-4, 1]);
    xlabel(ax,'X Position'); ylabel(ax,'Y Position');
    title(ax,'Interactive EMA Vocal Tract Positions');

    % Initial scatter and text
    scatterHandle = scatter(ax,sensorNeutral(:,1), sensorNeutral(:,2),80,'filled','k');
    textHandles = text(ax,sensorNeutral(:,1)+0.3, sensorNeutral(:,2)+0.3, sensorLabels,...
                       'FontSize',12,'FontWeight','bold');

    % Sensor groups indices
    jawIdx = ismember(sensorLabels,'JAW');
    lipIdxs = ismember(sensorLabels,{'UL','LL'});
    tongueIdxs = ismember(sensorLabels,{'T3','T2','T1'});

    % Callback nested function
    function updatePositions(~,~)
        jawVal = sliders(1).Value;
        lipVals = [sliders(2).Value sliders(3).Value sliders(4).Value];
        tongueVals = [sliders(5).Value sliders(6).Value sliders(7).Value];

        PVals = [tongueVals, jawVal, lipVals]';
        newPositions = reshape(PC * PVals, 2, [])' +  sensorNeutral; 
        

        % % Update Jaw
        % jaw_direction = jawCoeff(:,1)';
        % currentJawPos = sensorNeutral(jawIdx,:) + jawVal * jaw_direction;
        % 
        % % Update Lip (dependent on Jaw)
        % lipOffsets = reshape(lipCoeff(:,1:3)*lipVals',2,[])';
        % lipNeutralRelative = sensorNeutral(lipIdxs,:) - sensorNeutral(jawIdx,:);
        % % lipNeutralRelative(1,:) = lipNeutralRelative(1, :) + sensorNeutral(jawIdx,:); 
        % currentLipPos = currentJawPos + lipNeutralRelative + lipOffsets;
        % currentLipPos(1, :) = sensorNeutral(ismember(sensorLabels,{'UL'}), :) + lipOffsets(1,:);
        % % currentLipPos(1,:) = currentLipPos(1,:) - currentJawPos; % UL doesn't move with Jaw
        % 
        % % Update Tongue (dependent on Jaw)
        % tongueOffsets = reshape(tongueCoeff(:,1:3)*tongueVals',2,[])';
        % tongueNeutralRelative = sensorNeutral(tongueIdxs,:) - sensorNeutral(jawIdx,:);
        % currentTonguePos = currentJawPos + tongueNeutralRelative + tongueOffsets;
        % 
        % % Update positions array
        % newPositions = sensorNeutral;
        % newPositions(jawIdx,:) = currentJawPos;
        % newPositions(lipIdxs,:) = currentLipPos;
        % newPositions(tongueIdxs,:) = currentTonguePos;

        % Update scatter and labels
        scatterHandle.XData = newPositions(:,1);
        scatterHandle.YData = newPositions(:,2);

        for tIdx = 1:numel(textHandles)
            textHandles(tIdx).Position = [newPositions(tIdx,1)+0.3, newPositions(tIdx,2)+0.3, 0];
        end
    end

    % Initial call
    updatePositions();
end




%% --- Interactive EMA Vocal Tract Control GUI (corrected version) ---
% interactiveEMA_GUI(sensorNeutral, sensorLabels, jawCoeff, lipCoeff, tongueCoeff);


function interactiveEMA_GUI(sensorNeutral, sensorLabels, jawCoeff, lipCoeff, tongueCoeff)
    % GUI for EMA with Jaw-dependent articulator positions

    % Create figure
    fig = figure('Name','Interactive Vocal Tract Control',...
                 'NumberTitle','off','Position',[100 100 1200 600]);

    % Slider initial values
    slider_labels = {'Jaw Opening','Lip PC1','Lip PC2','Lip PC3',...
                     'Tongue PC1','Tongue PC2','Tongue PC3'};
    sliders = gobjects(1,7);
    slider_range = [-3 3];

    % Create sliders and labels
    for i=1:7
        sliders(i) = uicontrol('Parent',fig,'Style','slider',...
                     'Min',slider_range(1),'Max',slider_range(2),'Value',0,...
                     'Units','normalized','Position',[0.05, 0.9-0.1*i, 0.3, 0.04],...
                     'Callback', @updatePositions);

        uicontrol('Parent',fig,'Style','text','Units','normalized',...
                  'Position',[0.05, 0.94-0.1*i, 0.3, 0.03],'String',slider_labels{i},...
                  'FontSize',10,'HorizontalAlignment','left');
    end

    % Axes setup
    ax = axes('Parent',fig,'Position',[0.4 0.1 0.55 0.8]);
    hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
    xlim(ax,[-2, 6]); ylim(ax,[-4, 1]);
    xlabel(ax,'X Position'); ylabel(ax,'Y Position');
    title(ax,'Interactive EMA Vocal Tract Positions');

    % Initial scatter and text
    scatterHandle = scatter(ax,sensorNeutral(:,1), sensorNeutral(:,2),80,'filled','k');
    textHandles = text(ax,sensorNeutral(:,1)+0.3, sensorNeutral(:,2)+0.3, sensorLabels,...
                       'FontSize',12,'FontWeight','bold');

    % Sensor groups indices
    jawIdx = ismember(sensorLabels,'JAW');
    lipIdxs = ismember(sensorLabels,{'UL','LL'});
    tongueIdxs = ismember(sensorLabels,{'T3','T2','T1'});

    % Callback nested function
    function updatePositions(~,~)
        jawVal = sliders(1).Value;
        lipVals = [sliders(2).Value sliders(3).Value sliders(4).Value];
        tongueVals = [sliders(5).Value sliders(6).Value sliders(7).Value];

    

        % Update Jaw
        jaw_direction = jawCoeff(:,1)';
        currentJawPos = sensorNeutral(jawIdx,:) + jawVal * jaw_direction;

        % Update Lip (dependent on Jaw)
        lipOffsets = reshape(lipCoeff(:,1:3)*lipVals',2,[])';
        lipNeutralRelative = sensorNeutral(lipIdxs,:) - sensorNeutral(jawIdx,:);
        % lipNeutralRelative(1,:) = lipNeutralRelative(1, :) + sensorNeutral(jawIdx,:); 
        currentLipPos = currentJawPos + lipNeutralRelative + lipOffsets;
        currentLipPos(1, :) = sensorNeutral(ismember(sensorLabels,{'UL'}), :) + lipOffsets(1,:);
        % currentLipPos(1,:) = currentLipPos(1,:) - currentJawPos; % UL doesn't move with Jaw

        % Update Tongue (dependent on Jaw)
        tongueOffsets = reshape(tongueCoeff(:,1:3)*tongueVals',2,[])';
        tongueNeutralRelative = sensorNeutral(tongueIdxs,:) - sensorNeutral(jawIdx,:);
        currentTonguePos = currentJawPos + tongueNeutralRelative + tongueOffsets;

        % Update positions array
        newPositions = sensorNeutral;
        newPositions(jawIdx,:) = currentJawPos;
        newPositions(lipIdxs,:) = currentLipPos;
        newPositions(tongueIdxs,:) = currentTonguePos;

        % Update scatter and labels
        scatterHandle.XData = newPositions(:,1);
        scatterHandle.YData = newPositions(:,2);

        for tIdx = 1:numel(textHandles)
            textHandles(tIdx).Position = [newPositions(tIdx,1)+0.3, newPositions(tIdx,2)+0.3, 0];
        end
    end

    % Initial call
    updatePositions();
end


%% Try to convert the above function into matrix format 








