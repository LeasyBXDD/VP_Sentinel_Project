# python ./VCadv/inference.py --load "./checkpoints/again/c4s/again-c4s_100000.pth" --source "../lib/wav48/p225/p225_001.wav" --target "../lib/wav48/p226/p226_001.wav" --output "./output"
import subprocess

# 定义要运行的脚本的路径
script1 = "D:/AAA/lab/VP_Sentinel_Project/Sample_transformation/main.py"
script2 = "D:/AAA/lab/VP_Sentinel_Project/DeepSpeaker_TDNN/tdnnpro2.py"

# 使用 subprocess 运行第一个脚本
print("Running script 1...")
result1 = subprocess.run(["D:/AAA/lab/VP_Sentinel_Project/venv/Scripts/python", script1], capture_output=True, text=True)
print("Output of script 1:")
print(result1.stdout)
print("Error of script 1:")
print(result1.stderr)

# 使用 subprocess 运行第二个脚本
print("Running script 2...")
result2 = subprocess.run(["D:/AAA/lab/VP_Sentinel_Project/venv/Scripts/python", script2], capture_output=True, text=True)
print("Output of script 2:")
print(result2.stdout)
print("Error of script 2:")
print(result2.stderr)