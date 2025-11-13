import os
import cv2
import glob

# 设置输入和输出路径
frame_input_dir = './test/frames'
results_output_dir = './test/results'

# 确保输出目录存在
os.makedirs(results_output_dir, exist_ok=True)

def combine_frames_to_video(frame_dir, output_video_path, fps=30):
    """将帧合并为视频。"""
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_files:
        print(f"没有找到帧文件: {frame_dir}")
        return

    # 读取第一帧以获取视频分辨率
    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"完成: 合并视频 -> {output_video_path}")

def main():
    frame_dirs = sorted(glob.glob(os.path.join(frame_input_dir, "frames-*-UWMamba")))
    
    for frame_dir in frame_dirs:
        video_name = os.path.basename(frame_dir).split('-')[1]
        output_video_path = os.path.join(results_output_dir, f"processed-{video_name}-UWMamba.mp4")
        print(f"正在合并视频: {frame_dir}")
        combine_frames_to_video(frame_dir, output_video_path)

if __name__ == "__main__":
    main()
