classdef helperFeaturePointManager < handle
    %helperFeaturePointManager helper utility to 
    %   - detect new key points whenever number of tracked points go below a certain threshold mentioned in params
    %   - assign unique id for each key point, store key point tracks or 2D-2D key point correspondenses 
    %   - triangulate new 3D points from tracks whenver number tracked 3D points in current frame go below a certain threshold mentioned in params
    %   - store 3D points and 3D-2D correspondenses
    %   - manage sliding window
    %
    %   This is an example helper function that is subject to change or removal 
    %   in future releases.
    
    %   Copyright 2022-2023 The MathWorks, Inc.

    properties
        params
        % newly tracked key point unique ids that are not yet triangulated
        newPointIDs
        % camera intrinsics
        intrinsics

        % All frame key point observations
        AllObservations
        % All frame key point unique ids
        AllIds
        AllTriangulated

        % number of key points detected until now
        uniqueKeyPointCount

        % number of frames each key point is observed
        keyPointTrackCount

        slidingWindowViewIDs
        % Current sliding window index. can have a maximum value of window
        % size.
        currentSlidingWindowIndex

        % id of the current view
        currentViewID
        % id of the oldest view with new untriangulated tracks
        lastNewPointViewID

        % Triangulated 3-D point locations
        xyzPoints

        % first obsereved view of each 3-D point
        xyzStartView

        % ids of the valid triangulated 3D points
        xyzValIds

        % When data is collected with camera being stationary for a few
        % seconds like popular VIO datasets, set this to true. By default
        % we assume static camera at the start for good IMU bias estimation.
        noMovementAtStart

        % Some 3D points are triangulated using initial SFM when true 
        InitialMappingSuccessful

        % Key-point detector function handle
        DetectorFunc

        % Triangulation status of each uniqe key point. Although key points
        % are tracked in multiple frames a unique id is assigned to each
        % unique key point whenever detected.
        isTriangulated
        % Minimum number of triangulated point tracks in current view. If
        % tracked 3D point count in current view goes belowe this number
        % them new key points will be created and new 3D point
        % triangulation will be attempted.
        triangulatedThreshold
        % window state 
        %   - isWindowFull
        %   - isEnoughParallax
        %   - isFirstFewViews - still map not initialized and processing
        %     first few views 
        windowState

        % Status of each frame key frame or not
        isKeyFrame
    end

    methods
        function obj = helperFeaturePointManager(intrinsics, params, maxFrames, maxLandmarks)
            %helperFeaturePointManager constructor

            narginchk(2,4);

            if nargin < 3
                % by default assume maximum number of frames and landmarks for initializing the storage array variables 
                maxFrames = 5000;
                maxLandmarks = 5000;
            end
            % initialize and pre-allocate
            obj.intrinsics = intrinsics;
            obj.params = params;
            obj.uniqueKeyPointCount = 1;
            obj.AllObservations = cell(1,maxFrames);
            obj.AllIds = cell(1,maxFrames);
            obj.AllTriangulated = repmat({false(0,1)},1,maxLandmarks);
            obj.isTriangulated = false(maxLandmarks,1);
            obj.currentViewID = -1;
            obj.newPointIDs = [];
            obj.xyzPoints = [];
            obj.xyzStartView = [];
            obj.lastNewPointViewID = -1;
            obj.currentSlidingWindowIndex = 0;
            obj.slidingWindowViewIDs = zeros(params.windowSize,1);
            obj.xyzValIds = [];
            % by default assume no movement at the start of data aquisition
            % for a few seconds
            obj.noMovementAtStart = true;
            obj.InitialMappingSuccessful = false;
            % by default use Shi-Thomsi corner point detector
            obj.DetectorFunc = @(grayImage)detectMinEigenFeatures(grayImage, "MinQuality", 0.01, "FilterSize", 3);
            obj.triangulatedThreshold = 60;
            obj.windowState = struct('isEnoughParallax', false, 'isWindowFull', false, 'isFirstFewViews', false);
            obj.isKeyFrame = false(maxFrames,1);
        end

        function [rmF, windowState] = updateSlidingWindow(obj,I,currPointsTracked,validIdx,viewId)
            %updateSlidingWindow update the sliding window after last view
            %   tracking. If the number of frames in sliding window
            %   is greater than window size after new frame addition remove
            %   one frame.

            windowState = obj.windowState;
            if obj.currentSlidingWindowIndex ==0
                % the very first frame. accept it right away. 
                obj.currentSlidingWindowIndex = obj.currentSlidingWindowIndex + 1;
                rmF = -1;
                obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex) = viewId;
                return;
            end
            
            % find false tracks or outlier matches to discard them
            psIdx=helperWithinImage(currPointsTracked,size(I));
            v1 = validIdx&psIdx;
            inlF = [];            
            for k = 1:obj.params.F_loop
                [~,inlFf]=estimateFundamentalMatrix(obj.AllObservations{viewId-1}(v1,:),currPointsTracked(v1,:),...
                    'Method', 'RANSAC', 'NumTrials', obj.params.F_Iterations, ...
                    'Confidence',obj.params.F_Confidence,'DistanceThreshold', obj.params.F_Threshold);
                if length(find(inlFf)) > length(find(inlF))
                    inlF = inlFf;
                end
            end
            inlFf = false(size(v1));
            inlFf(v1) = inlF;
            v1 = v1 & inlFf;
            % update feature tracks
            obj.AllObservations{viewId} = currPointsTracked(v1,:);
            obj.AllTriangulated{viewId} = obj.AllTriangulated{max(1,viewId-1)}(v1,:);
            pIds = obj.AllIds{max(1,viewId-1)}(v1,2);
            obj.keyPointTrackCount(pIds) = obj.keyPointTrackCount(pIds) + 1;
            obj.AllIds{viewId} = [viewId*ones(size(pIds)),pIds];
            obj.currentViewID = viewId;
            
            % remove 1 frame if the sliding window is full to accommodate
            % the current frame.
            rmF = obj.slidingWindowViewIDs(1);
            noMoveWindow = 0.5; % between 0 and 1. 
            if  (obj.currentSlidingWindowIndex < floor(obj.params.windowSize*noMoveWindow) && ...
                    obj.noMovementAtStart) || (obj.currentSlidingWindowIndex < 2)
                    % accept a few very first frames without any additional
                    % processing if there is not movement at the start.
                    % These frames will help in bias estimation.
                    
                    if (obj.currentSlidingWindowIndex == (floor(obj.params.windowSize*noMoveWindow)-1) && ...
                            obj.noMovementAtStart) || (~obj.noMovementAtStart && obj.currentSlidingWindowIndex == 1)
                        [~,IA,IB] = intersect(obj.AllIds{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)}(:,2),obj.AllIds{viewId}(:,2),'legacy');
                        m1 = obj.AllObservations{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)}(IA,:);
                        m2 = obj.AllObservations{viewId}(IB,:);
                        [~,isKf] = helperQuickCheckParallax(m1,m2,obj.params.keyFrameParallax);
                        if isKf
                            % last frame is a key frame, then increment sliding
                            % window until there is space and accept current frame
                            obj.isKeyFrame(viewId) = true;
                            obj.currentSlidingWindowIndex = obj.currentSlidingWindowIndex + 1;
                            obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex) = viewId;
                            rmF = -2;
                            windowState.isEnoughParallax = true;
                        else
                            rmF = viewId;
                        end
                    else
                        obj.currentSlidingWindowIndex = obj.currentSlidingWindowIndex + 1;
                        obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex) = viewId;
                        rmF = -3;
                        windowState.isFirstFewViews = true;
                    end
            elseif (obj.currentSlidingWindowIndex >= floor(obj.params.windowSize*noMoveWindow) && obj.currentSlidingWindowIndex < obj.params.windowSize && ...
                    obj.noMovementAtStart) || (obj.currentSlidingWindowIndex < obj.params.windowSize && ~obj.noMovementAtStart)
                % accept current frame and remove last frame if there isn't
                % enough parallax between last frame and last key frame.
                [~,IA,IB] = intersect(obj.AllIds{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex-1)}(:,2),obj.AllIds{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)}(:,2),'legacy');
                 m1 = obj.AllObservations{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex-1)}(IA,:);
                 m2 = obj.AllObservations{obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)}(IB,:);
                 [~,isKf] = helperQuickCheckParallax(m1,m2,obj.params.keyFrameParallax);

                if isKf || obj.isKeyFrame(obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex))
                    % last frame is a key frame, then increment sliding
                    % window until there is space and accept current frame
                    obj.isKeyFrame(obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex)) = true;
                    obj.currentSlidingWindowIndex = obj.currentSlidingWindowIndex + 1;
                    obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex) = viewId;
                    rmF = -2;
                    windowState.isEnoughParallax = true;
                else
                    % remove last frame if its not a key frame
                    rmF = obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex);
                    obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex) = viewId;
                end
            else
                % window is full
                windowState.isWindowFull = true;
                [~,IA,IB] = intersect(obj.AllIds{obj.slidingWindowViewIDs(end-1)}(:,2),obj.AllIds{obj.slidingWindowViewIDs(end)}(:,2),'legacy');
                m1 = obj.AllObservations{obj.slidingWindowViewIDs(end-1)}(IA,:);
                m2 = obj.AllObservations{obj.slidingWindowViewIDs(end)}(IB,:);
                [~,isKf] = helperQuickCheckParallax(m1,m2,obj.params.keyFrameParallax);
                if ~isKf || ~obj.isKeyFrame(obj.slidingWindowViewIDs(end))
                    % accept current frame and remove last frame since it's
                    % not a key frame.
                    rmF = obj.slidingWindowViewIDs(end);
                    obj.slidingWindowViewIDs(end) = viewId;
                else
                    % remove first frame in sliding window since the last
                    % frame is a key frame.
                    obj.slidingWindowViewIDs = [obj.slidingWindowViewIDs(2:end);viewId];
                    windowState.isEnoughParallax = true;
                    obj.isKeyFrame(obj.slidingWindowViewIDs(end-1)) = true;
                end
            end
            
        end

        function currPoints = createNewFeaturePoints(obj,I)
            %createNewFeaturePoints create new feature points and return
            %   all points in current frame tracked and new 

            if obj.currentViewID < 0
                % this is the first time create points is called. adjust
                % the view id accordingly
                obj.currentViewID = 1;
                obj.isKeyFrame(1) = true;
            end
            % get key points tracked in current frame 
            currPointsTracked = obj.AllObservations{obj.currentViewID};
            if size(currPointsTracked,1) < obj.params.numTrackedThresh || ...
                    (length(find(obj.isTriangulated(obj.AllIds{obj.currentViewID}(:,2)))) < obj.triangulatedThreshold && ...
                    obj.InitialMappingSuccessful && (obj.params.maxPointsToTrack-size(currPointsTracked,1) > 4))
                % create new key points only when current tracked key points go
                % below a threshold or the number of triangulated point
                % tracks go below a certain threshold
                cp = obj.DetectorFunc(I);
                cp = helperSelectNewKeyPointsUniformly(currPointsTracked,cp,30,size(I),obj.params.maxPointsToTrack);

                % assign unique id for each new detected corner
                newPointUniqueIds = obj.uniqueKeyPointCount + (0:(size(cp,1)-1))';

                obj.uniqueKeyPointCount = obj.uniqueKeyPointCount + size(cp,1);
                % compute all points in current frame tracked and new
                currPoints = [currPointsTracked;double(cp.Location)];
                % store new points and unique ids
                obj.AllObservations{obj.currentViewID} = currPoints;
                obj.AllIds{obj.currentViewID} = [obj.AllIds{obj.currentViewID};[obj.currentViewID*ones(size(newPointUniqueIds)),newPointUniqueIds]];
                obj.AllTriangulated{obj.currentViewID} = [obj.AllTriangulated{obj.currentViewID};false(size(newPointUniqueIds))];
                if isempty(obj.newPointIDs)
                    obj.lastNewPointViewID = obj.currentViewID;
                end
                % update the new point ids that are not triangulated
                obj.newPointIDs = [obj.newPointIDs;newPointUniqueIds];
                % update the tracked count for each new point
                obj.keyPointTrackCount = [obj.keyPointTrackCount;ones(size(newPointUniqueIds))];
                % update the start view for each new key point
                obj.xyzStartView = [obj.xyzStartView;ones(size(newPointUniqueIds))*obj.currentViewID];
                obj.isKeyFrame(obj.currentViewID) = true;
            end
        end

        function [newXYZ,newwXYZUniqueIds,allNewPointViews,allNewObservations] = triangulateNew3DPoints(obj, vSet)
            %triangulateNew3DPoints triangulates new 3D points from 2D-2D
            %   key point correspondenses or tracks and key frame poses
            %   stored in view set. The triangulation alway happens only
            %   between last 2 key frames in the sliding window.

            % get all new key point tracks that are atleast seen in 2
            % frames
            goodNTrIdx = obj.keyPointTrackCount(obj.newPointIDs) > 1;
            allNewPointViews = [];
            allNewObservations = {};
            if isempty(goodNTrIdx)
                obj.newPointIDs = [];
                newXYZ = [];
                newwXYZUniqueIds = [];
                return;
            end

            newwXYZUniqueIds = obj.newPointIDs(goodNTrIdx);
            % get ids of last 2 key frames
            lastKeyFrameId1 = obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex-1);
            lastKeyFrameId2 = obj.slidingWindowViewIDs(obj.currentSlidingWindowIndex);

            % get 2D-2D correspondense between last 2 key frames that are
            % not yet used for triangulating 3D points
            [trIdsO,pIdxO,pTRIdxo] = intersect(obj.AllIds{lastKeyFrameId1}(:,2),newwXYZUniqueIds);
            trNew = true(size(newwXYZUniqueIds,1),1);
            trNew(pTRIdxo) = false;
            tV = lastKeyFrameId1;
            [trIdsPCO,pIdxpO,pIdxpcO] = intersect(obj.AllIds{lastKeyFrameId2}(:,2),trIdsO);
            m1 = obj.AllObservations{lastKeyFrameId1}(pIdxO(pIdxpcO),:);
            m2 = obj.AllObservations{lastKeyFrameId2}(pIdxpO,:);

            % check parallax between last 2 key frames
            [~,isEn] = helperQuickCheckParallax(m1,m2,obj.params.triangulateParallax);
            
            if isEn || (length(find(obj.isTriangulated(obj.AllIds{obj.currentViewID}(:,2)))) < obj.triangulatedThreshold && obj.InitialMappingSuccessful && ~isempty(m1))
                % when enough parallax between key frames is found
                % triangulate 3D points
                camMatrix1 = cameraProjection(obj.intrinsics, pose2extr(poses(vSet,tV).AbsolutePose));
                camMatrix2 = cameraProjection(obj.intrinsics, pose2extr(poses(vSet,lastKeyFrameId2).AbsolutePose));
                [newXYZ, ~, isInFront] = triangulate(m1, ...
                    m2, camMatrix1, camMatrix2);
                % Filter points by view direction
                vI  = isInFront;
                newXYZ = newXYZ(vI,:);

                trIdsPC1 = trIdsPCO(vI);
                obj.isTriangulated(trIdsPC1) = true;
                [~,pIc, pIc1] = intersect(obj.AllIds{obj.currentViewID}(:,2),trIdsPCO);
                
                pIdxpci = false(size(obj.AllIds{obj.currentViewID},1),1);
                pIdxpci(pIc) = vI(pIc1);
                obj.AllTriangulated{obj.currentViewID}(pIdxpci) = true;
            else
                obj.newPointIDs = newwXYZUniqueIds;
                newXYZ = [];
                newwXYZUniqueIds = [];
                return;
            end
            % store new 3D points
            obj.xyzPoints = [obj.xyzPoints;zeros(size(obj.newPointIDs,1),3)];
            obj.xyzPoints(trIdsPC1,:) = newXYZ;
            obj.xyzValIds = [obj.xyzValIds;trIdsPC1];

            if ~obj.InitialMappingSuccessful
                slvId  = 1;
                obj.InitialMappingSuccessful = true;
            else
                slvId = find(obj.slidingWindowViewIDs >= min(obj.xyzStartView(trIdsPC1)),1,'first');
            end
            % compute all observations of newly triangulated points 
            allNewPointViews = [];
            allNewObservations = {};
            for k = slvId:obj.currentSlidingWindowIndex
                k2 = obj.slidingWindowViewIDs(k);
                [~,pIdxx] = intersect(obj.AllIds{k2}(:,2),trIdsPC1);
                if ~isempty(pIdxx)
                    allNewPointViews(end+1) = k2; %#ok
                    allNewObservations{end+1} = [obj.AllIds{k2}(pIdxx,:),obj.AllObservations{k2}(pIdxx,:)];%#ok
                end
            end

            obj.newPointIDs = newwXYZUniqueIds(trNew,:);
            newwXYZUniqueIds = trIdsPC1;
        end

        function [matches1,matches2] = get2DCorrespondensesBetweenViews(obj,id1,id2)
            %get2DCorrespondensesBetweenViews returns 2D-2D correspondenses
            % between specified view ids id1 and id2

            [~,IA,IB] = intersect(obj.AllIds{id1}(:,2),obj.AllIds{id2}(:,2),'legacy');
            matches1 = obj.AllObservations{id1}(IA,:);
            matches2 = obj.AllObservations{id2}(IB,:);
        end

        function [keyPoints,uniquePointIds,isTriangulated] = getKeyPointsInView(obj, viewId)
            %getKeyPointsInView returns key points seen in specified view,
            %   unique id of each keypoint and status of weather a 3D point
            %   correspondense exists for it.

            keyPoints = obj.AllObservations{viewId};
            uniquePointIds = obj.AllIds{viewId}(:,2);
            isTriangulated = obj.isTriangulated(uniquePointIds);
        end

        function ids = getPointIdsInViews(obj, viewIds)

            allI = vertcat(obj.AllIds{viewIds});
            uI = unique(allI(:,2));
            isT = obj.isTriangulated(uI);
            ids = uI(isT);

        end

        function setKeyPointValidityInView(obj, viewId, validity)
            %setKeyPointValidityInView set key point validity in spefified 

            obj.AllObservations{viewId} = obj.AllObservations{viewId}(validity,:);
            obj.AllIds{viewId} = obj.AllIds{viewId}(validity,:);
            obj.AllTriangulated{viewId} = obj.AllTriangulated{viewId}(validity);
        end

        function [xyz,ids] = getXYZPoints(obj,ids)
            %getXYZPoints get xyz points specified by ids. if ids is not
            %   specified returns all valid 3D points and corresponding ids

            if nargin < 2
                ids = obj.xyzValIds;
            end
            xyz = obj.xyzPoints(ids,:);
            ids = obj.xyzValIds;
        end

        function setXYZPoints(obj,xyz,ids)
            %setXYZPoints set specified xyz points with new values

            if nargin < 3
                ids = obj.xyzValIds;
            end
            obj.xyzPoints(ids,:) = xyz;
        end

        function swIDs = getSlidingWindowIDs(obj)
            %getSlidingWindowIDs

            swIDs = obj.slidingWindowViewIDs(1:obj.currentSlidingWindowIndex);
        end
    end

end

% local functions
function isWithinImage = helperWithinImage(points, imageSize)
%helperWithinImage helper function to check validity of points

isWithinImage = (points(:,1) >= 1) & (points(:,1) <= (imageSize(2))) & (points(:,2) >= 1) & (points(:,2) <= (imageSize(1)));
end

function [avg,status] = helperQuickCheckParallax(matches1,matches2,parallaxThreshold)
%helperQuickCheckParallax helper to compute parallax between 2 camera views.
%   m1 and m2 are matched points (2D-2D correspondenses) between the views.

A = matches1-matches2;
avg = sum(sqrt(sum(A.*A,2)))/size(A,1);
% return true if the computed parallax is greater than threshold
status = avg > parallaxThreshold;
end

function newCornersAwayFromExisting = helperSelectNewKeyPointsUniformly(currentTrackedCorners,newCorners,minDist,imageSize,maxCornerCount)
%helperSelectNewKeyPointsUniformly helper function to select new key points
%   uniformly spaced and away from the existing tracked key points.

% compute image mask that's only true when the pixel is away from an
% existing corner by a specified distance
imageMask = false(imageSize+(2*minDist));
if ~isempty(currentTrackedCorners)
    [x,y] = meshgrid(-minDist:minDist);
    ind = (x(:).^2 + y(:).^2 <= (minDist + 0.75)^2);
    x = x(ind)';
    y = y(ind)';
    cx = round(currentTrackedCorners(:,1)) + x + minDist;
    cy = round(currentTrackedCorners(:,2)) + y + minDist;
    cxy = ((cx-1)*(imageSize(1) + 2*minDist)) + cy;
    imageMask(cxy) = true;
end
% apply the mask to compute indices of corners away from existing corners
newGoodCornerInd = (round(newCorners.Location(:,1)) + minDist -1)*(imageSize(1) + 2*minDist) + round(newCorners.Location(:,2)) + minDist;
newCornersAwayFromExistingAll = newCorners(~imageMask(newGoodCornerInd));
numNewCornersToSelect = maxCornerCount - size(currentTrackedCorners,1);
% select uniformly from good corners
newCornersAwayFromExisting = selectUniform(newCornersAwayFromExistingAll,numNewCornersToSelect,imageSize);
end