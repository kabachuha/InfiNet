import os
import math
import sys
import argparse
import cv2
import csv
from tqdm import tqdm

def chop_video(video_path: str, depth: int, max_frames: int) -> None:
    """
    Chop a video into subsets based on specified depth and maximum number of frames per subset.

    Args:
        video_path (str): path to input video file.
        depth (int): number of depth levels to split the video into subsets.
        max_frames (int): maximum number of frames per subset video file.

    Returns:
        None: This function doesn't return anything. It saves the video subsets and txt files in the same directory as the input video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")


    # Initialize the CSV data storage
    csv_data = [["Clip Name", "Frame Number"]]
    for i in range(depth):
        csv_data[0].append(f"Depth {i}")

    # Create a list to store "P" markers for each depth and frame number
    p_marker_list = [{} for _ in range(depth)]

    # Open input video file and get its properties
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame skip value for each depth level
    frame_skip_at_depth = []
    for i in range(depth):
        frame_skip = math.log2(total_frames) * i
        frame_skip_at_depth.append(int(frame_skip))
    frame_skip_at_depth.reverse()

    # Read all frames of the input video into memory
    video_frames = []
    for _ in tqdm(range(total_frames), desc='Reading video frames'):
        ret, frame = video.read()
        if ret:
            video_frames.append(frame)
    # Iterate over each depth level and create video subsets and txt files
    dir_name = ""
    curr_depth = 0
    for frame_skip in frame_skip_at_depth:
        if frame_skip != 0:
            work_frames = total_frames - total_frames % frame_skip
            subset_frames = int(work_frames / frame_skip)
        else:
            subset_frames = int(total_frames)

        num_videos = math.ceil(subset_frames / max_frames)
        video_count = 0
        if dir_name == "":
            dir_name = f"depth_{curr_depth}"
        else:
            dir_name = os.path.join(dir_name, f"depth_{curr_depth}")
        os.makedirs(dir_name, exist_ok=True)

        # Iterate over each video subset within the depth level
        for i in tqdm(range(num_videos), desc=f'Depth {curr_depth}'):
            # Define the output video file name and properties
            output_filename = f"{dir_name}/subset_{video_count}.mp4"
            height, width, _ = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

            frame_skip = 1 if frame_skip == 0 else frame_skip

            # Determine the frame range for the current video subset
            frame_index = i * max_frames * frame_skip
            end_index = min(frame_index + max_frames * frame_skip, subset_frames * frame_skip, total_frames)

            # Write the frames for the current video subset to the output file
            for j in tqdm(range(frame_index, end_index, frame_skip), desc=f'Subset {video_count}'):
                out.write(video_frames[j])
                p_marker_list[curr_depth][j] = "P"

            out.release()

            # Save a txt file for the current video subset
            with open(f"{dir_name}/subset_{video_count}.txt", "w") as f:
                f.write(f"Subset {video_count}")

            video_count += 1

        curr_depth += 1

    video.release()



    # Generate the CSV data based on the "P" marker list
    for frame_num in range(total_frames):
        csv_row = [""] * (2 + depth)
        csv_row[1] = frame_num
        for d in range(depth):
            if frame_num in p_marker_list[d]:
                csv_row[2 + d] = "P"
        csv_data.append(csv_row)

    # Write the CSV data to a file
    with open("chopped_video_info.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

def closest_divisible_integer(x, n):
    remainder = x % n
    if remainder <= n/2:
        return x - remainder
    else:
        return x - n + remainder

def main():
    parser = argparse.ArgumentParser(description="Chop a video file into subsets of frames.")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("depth", type=int, help="Desired depth level.")
    parser.add_argument("max_frames", type=int, help="Maximum frames in a subset at the highest depth level.")
    args = parser.parse_args()
    chop_video(args.video_file, args.depth, args.max_frames)

if __name__ == "__main__":
    main()
