# 视频流AI智能识别系统

一个结合YOLO目标检测和视觉大模型的实时视频流处理系统，支持RTSP推流、实时检测和AI智能描述。

##  项目简介

本项目实现了一个完整的视频流处理管线：
- 从RTSP服务器读取视频流
- 使用YOLOv8进行实时目标检测
- 调用视觉大模型（如Qwen-VL）分析视频内容
- 将处理后的视频（含检测框和AI描述）推流到RTSP服务器

##  主要特性

- **实时目标检测**：基于YOLOv8，可检测80+种常见物体
- **AI智能描述**：自动分析视频内容并生成文字描述
- **RTSP推流**：处理后的视频可重新推流，支持多客户端观看
- **异步处理**：AI调用不阻塞视频流，保证流畅性
- **性能优化**：
  - 自动缩放图片减少API传输时间
  - 可配置AI调用频率
  - 使用ultrafast编码降低延迟
- **可视化增强**：
  - 检测框标注
  - AI描述文字叠加
  - 实时统计信息显示

##  快速开始

### 环境要求

- Python 3.8+
- FFmpeg（用于推流）
- RTSP服务器（如EasyDarwin）

### 安装依赖

'''bash
# 安装Python包
pip install opencv-python ultralytics langchain-openai numpy

# 安装FFmpeg（Ubuntu/Debian）
sudo apt-get install ffmpeg

# 或macOS
brew install ffmpeg

# 或Windows（下载安装包）
# https://ffmpeg.org/download.html
配置说明
修改API密钥（必需）
'''

打开 3.py，替换以下内容：

python
openai_api_key="你的API密钥"
调整视频参数（可选）

python
INPUT_URL = "rtsp://127.0.0.1:25544/input"   # 输入流地址
OUTPUT_URL = "rtsp://127.0.0.1:25544/output" # 输出流地址
self.ai_frame_interval = 30  # AI调用间隔（帧数）
更换AI模型（可选）

python
model="Qwen/Qwen3-VL-8B-Instruct"  # 可替换为其他支持的多模态模型
运行
bash
python 3.py

# 使用示例

1. 准备视频源
假设你有一个RTSP视频源（如IP摄像头），或者使用EasyDarwin搭建测试环境：

bash
推一个测试视频到RTSP服务器
ffmpeg -re -i test.mp4 -f rtsp rtsp://127.0.0.1:25544/input

2. 运行处理程序
bash
python 3.py
程序启动后会显示：

输入/输出RTSP地址

实时处理窗口（按 'q' 退出）

AI分析结果打印到控制台

3. 观看处理后的视频流
使用任何支持RTSP的播放器观看：

bash
# VLC
vlc rtsp://127.0.0.1:25544/output

# FFplay
ffplay rtsp://127.0.0.1:25544/output

 # 应用场景
 
智能安防监控：实时检测可疑物体并生成警报描述

交通监控：分析车流量、识别违章行为

零售分析：统计客流、分析顾客行为

直播增强：为直播画面添加智能字幕和解说

工业检测：生产线质量监控和缺陷描述

 # 配置参数
参数	说明	默认值
ai_frame_interval	AI调用帧间隔	30
temperature	AI创造性（0-1）	0.7
max_tokens	AI回复最大长度	512
preset	编码速度	ultrafast
crf	视频质量（越小越好）	23
question_template	AI提问模板	"简要描述..."

 # 性能说明
 
处理延迟：约100-200ms（取决于硬件）

AI响应时间：2-5秒（取决于API和模型）

建议帧率：15-30fps

CPU占用：中等（YOLO检测占用较高）

GPU加速：YOLO支持CUDA加速

 # 常见问题
 
1. 无法打开RTSP流

确认RTSP服务器已启动

检查地址格式是否正确

验证防火墙设置

2. FFmpeg推流失败

确认FFmpeg已正确安装：ffmpeg -version

检查输出RTSP地址是否可写

尝试降低帧率或分辨率

3. AI调用失败

检查API密钥是否有效

确认网络能访问API地址

检查账户余额/配额

4. 内存占用过高

减小 ai_frame_interval 值

降低输入视频分辨率

添加帧队列限制

 项目结构
video
.
├── 3.py   # 主程序
├── README.md               # 项目文档


 # 扩展开发
 
更换YOLO模型

python
# 使用更大的模型提高精度
self.model = YOLO("yolov8x.pt")

# 或使用自己训练的模型
self.model = YOLO("custom_model.pt")

# 自定义AI提示词
python
self.question_template = """
请分析这张图片并回答：
1. 图片中有哪些物体？
2. 这些物体之间有什么关系？
3. 有什么需要注意的安全隐患？

"""
添加更多处理功能

python
def process_frame(self, frame):
    # 添加自定义处理
    frame = self.face_detection(frame)
    frame = self.license_plate_recognition(frame)
    # ... 其他处理

  
# 注意事项
API费用：调用视觉大模型API会产生费用，请合理设置调用频率

隐私保护：确保视频内容符合当地隐私法规

网络要求：需要稳定的网络连接访问API

硬件要求：建议至少4GB内存，支持CUDA的GPU可大幅提升性能

# 技术栈
视频处理：OpenCV, FFmpeg

目标检测：YOLOv8 (Ultralytics)

AI模型：LangChain + SiliconFlow API

# 实验结果
<img width="2316" height="725" alt="image" src="https://github.com/user-attachments/assets/5667eab5-7dd8-41f9-a50d-697ea21b115d" />

