import os
import sys
import argparse
import cv2
from tqdm import tqdm
from pathlib import Path

def move_the_files(init_path, L, depth):

    folder_dataset_path = init_path
    depth_name = init_path

    # complexity O(L^d), exponential; but it's expected for a tree-like structure
    for d in range(0, depth):
        depth_name = os.path.join(depth_name, f'depth_{d}')
        for j in range(L**(d-1) if d > 1 else 1):
            part_path = os.path.join(depth_name, f'part_{j}')
                # sample the text info for the next subset
            for i in range(L if d > 0 else 1):
                txt_path = os.path.join(part_path, f'subset_{i}.txt')
                
                # go to the subset for video frames sampling
                next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
                next_part_path = os.path.join(next_depth_name, f'part_{i}') # `i` cause we want to sample each corresponding *subset*

                # depths > 0 are *guaranteed* to have L videos in their part_j folders
                
                # now sampling each first frame at the next level
                L_frames = [read_first_frame(os.path.join(next_part_path, f'subset_{k}.mp4')) for k in range(L)]
                
                # write all the L sampled frames to an mp4 in the folder dataset
                write_as_video(os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i}.mp4')
                


def main():
    parser = argparse.ArgumentParser(description="Convert the chopped labeled tree-like data into a FolderDataset")
    parser.add_argument("video_file", help="Path to the video file.")
    parser.add_argument("--L", help="Num of splits on each level.")
    args = parser.parse_args()
    move_the_files(args.video_file, int(args.L))

if __name__ == "__main__":
    main()
    