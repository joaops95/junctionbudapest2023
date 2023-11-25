import cv2
import os
import random
import requests
import json
import base64
import threading
import argparse

def split_video_into_frames(video_path, output_folder, number_of_frames=5):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0

    while success:
        # cv2.imwrite(os.path.join(output_folder, f"frame{count}.jpg"), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1



    selected_frames = random.sample(range(count), number_of_frames)
    
    for frame in selected_frames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vidcap.read()
        cv2.imwrite(os.path.join(output_folder, f"frame{frame}.jpg"), image)


    selected_frame_paths = [os.path.join(output_folder, f"frame{frame}.jpg") for frame in selected_frames]
    
    return selected_frame_paths


def get_image_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string



def call_deepfake_detection_api(image_base64):
    """Call the deepfake detection API and return the response."""
    url = "https://aaronespasa-deepfake-detection.hf.space/api/predict"
    payload = {
        "data": [f"data:image/png;base64,{image_base64}", ""],
    }

    # print("Calling API...", payload)
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    # print("API Response:", response)
    return response.json()


def process_frames_with_threads(frame_paths):
    results = []  # List to store results from each thread
    threads = []

    for path in frame_paths:
        thread = threading.Thread(target=detect_deepfake, args=(path, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Check if any of the results is True (i.e., fake detected)
    if any(results):
        print("Fake detected")
    else:
        print("No fake detected")


def detect_deepfake(image_path, results):
    image_base64 = get_image_base64(image_path)

    # Call the API
    try:
        response = call_deepfake_detection_api(image_base64)
        confidences = response["data"][0]["confidences"]
        # print(f"Confidences for {image_path}:", confidences)
        fake_confidence = [c['confidence'] for c in confidences if c['label'] == 'fake'][0]
        # print(f"Fake Confidence for {image_path}:", fake_confidence)
        is_fake = fake_confidence > 0.5
        results.append(is_fake)
    except Exception as e:
        print(f"API call failed for {image_path}: {e}")
        results.append(False)  # Assuming false if API call fails


def main(video_path, output_folder, number_of_frames):
    selected_frames = split_video_into_frames(video_path, output_folder, number_of_frames)
    process_frames_with_threads(selected_frames)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deepfake detection in video frames')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_folder', type=str, default='frames', help='Path to the output folder')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to extract and analyze (default: 5)')

    args = parser.parse_args()
    main(args.video_path, args.output_folder, args.frames)
