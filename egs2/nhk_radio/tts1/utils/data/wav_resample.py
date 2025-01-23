import os
import subprocess

def resample_wavs(input_dir, output_dir, target_sample_rate=22050):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有 .wav 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # 调用 ffmpeg 重新采样
            cmd = [
                "ffmpeg", "-i", input_path,
                "-ar", str(target_sample_rate),
                output_path
            ]
            subprocess.run(cmd, check=True)
            print(f"Resampled: {input_path} -> {output_path}")

# 示例使用
input_dir = "/home/hsieh/espnet/egs2/nhk_radio/tts1/data/wav"  # 替换为你的输入文件夹路径
output_dir = "/home/hsieh/espnet/egs2/nhk_radio/tts1/data/wav_22050"  # 替换为你的输出文件夹路径
resample_wavs(input_dir, output_dir, target_sample_rate=22050)
