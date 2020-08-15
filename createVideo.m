function createVideo(USframesDir, VideoFileName, yyyy_1, mm_1, dd_1, hh_1, min_1, ss_1, yyyy_2, mm_2, dd_2, hh_2, min_2, ss_2)
%videoStartingTime = datetime(2018,3,19,14,41,18); %Y,M,D,H,MI,S
%videoEndingTime = datetime(2018,3,19,14,58,17); %Y,M,D,H,MI,S
%USframesDir = 'F:\git\PancreaticProject2\PancreaticProject\RF afablation adrenal gland\recordings\US\20180319T144830';

files = dir([USframesDir '/*png']);

videoStartingTime = datetime(yyyy_1, mm_1, dd_1, hh_1, min_1, ss_1); %Y,M,D,H,MI,S
videoEndingTime = datetime(yyyy_2, mm_2, dd_2, hh_2, min_2, ss_2); %Y,M,D,H,MI,S

numSeconds = seconds(videoEndingTime-videoStartingTime);
numFrames = size(files,1);

framesPerSecond = numFrames/numSeconds;

v = VideoWriter(VideoFileName);
v.FrameRate = framesPerSecond;

disp(['Frame rate: ' num2str(framesPerSecond)]);

open(v);
for i=1:numFrames
   
    frame = imread([USframesDir '/synchronized_Frame_' num2str(i) '.png']);
    writeVideo(v,frame);
end
close(v);

disp('Movie saved.');
end

