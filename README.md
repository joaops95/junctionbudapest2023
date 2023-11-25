
# Project Title

This project contains two modules:

## Face Pose Detection: 
This module detects face features and positions either from a live camera feed or from a provided video file.

## Deepfake Detection: 
This module analyzes images to determine if they are deepfakes.

Installation

First, ensure you have Python installed on your system. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

# Module 1: Face Pose Detection
Description
The Face Pose Detection module identifies and marks facial features and their positions in real-time using a camera or by processing a given video file.

Usage
To use this module, run the script with the following command-line arguments:

-p or --path: Specify the path to a video file.
-r or --record_path: Specify the path where the output video will be saved.


```bash
python face_pose_detection.py --path <path_to_video> --record_path <path_to_output_video>
```

If no arguments are provided, the module will default to using a live camera feed.

Example

```bash
python face_pose_detection.py --path input_video.mp4 --record_path output_video.avi
```

This command will process the video file 'input_video.mp4' and save the output video to 'output_video.avi'.

## Output

The module will output a video file with the detected facial features marked. And if the video is a deepfake or not, using the module from Module 2.


# Module 2: Deepfake Detection

## Description

The Deepfake Detection module extracts frames from a video, randomly selects a specified number of frames, and analyzes them to detect potential deepfakes.


```bash
python deepfake_detection_script.py <video_path> <output_folder> [--frames <number_of_frames>]
```
video_path: Path to the video file you want to analyze.
output_folder: Path to the folder where extracted frames will be saved.
--frames: Optional. Number of random frames to extract from the video for analysis. Default is 5.

## Example:

```bash
python deepfake_detection_script.py example_video.mp4 extracted_frames --frames 10
```

This command analyzes 'example_video.mp4', saves the extracted frames in 'extracted_frames', and analyzes 10 random frames from the video.

## Output:
The script will output the analysis result to the console, indicating whether a deepfake is detected in any of the analyzed frames.

