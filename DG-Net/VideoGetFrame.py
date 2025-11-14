import os
import cv2
import glob

# 设置输入和输出路径
test_dir = './video1102'
frame_output_dir = './video1102/Frames'

# 确保输出目录存在
os.makedirs(frame_output_dir, exist_ok=True)

def extract_frames(video_path, frame_dir):
    """从视频中提取帧并保存到指定目录。"""
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"完成: 提取 {frame_count} 帧 -> {frame_dir}")
    return frame_count

def main():
    video_files = sorted(glob.glob(os.path.join(test_dir, "*.mp4")) +
                         glob.glob(os.path.join(test_dir, "*.avi")))
    
    for video_file in video_files:
        video_name = os.path.basename(video_file).split('.')[0]
        frame_dir = os.path.join(frame_output_dir, f"frames-{video_name}")
        print(f"正在处理视频: {video_file}")
        extract_frames(video_file, frame_dir)

if __name__ == "__main__":
    main()
