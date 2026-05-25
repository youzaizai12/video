import cv2
import subprocess
import base64
import threading
import queue
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time
import numpy as np


class VideoAIProcessor:
    def __init__(self, rtsp_input_url, rtsp_output_url, frame_width=1280, frame_height=720):
        # YOLO模型
        self.model = YOLO("yolov8n.pt")

        # 视觉大模型配置
        self.vision_model = ChatOpenAI(
            openai_api_key="sk-iygzhbgmrqdosjzxmyblekyqembpkvknubmefbtlqcckctyk",
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen3-VL-8B-Instruct",
            temperature=0.7,
            max_tokens=512  # 减少token数量加快响应
        )

        # 视频参数
        self.rtsp_input = rtsp_input_url
        self.rtsp_output = rtsp_output_url
        self.frame_width = frame_width
        self.frame_height = frame_height

        # AI调用控制
        self.ai_frame_interval = 30  # 每30帧调用一次AI
        self.frame_count = 0
        self.latest_ai_description = "等待AI分析..."
        self.ai_response_queue = queue.Queue()

        # 运行标志
        self.running = True

        # 问题模板
        self.question_template = "简要描述这张图片中的主要内容，包括物体、人物和场景。"

    def frame_to_base64(self, frame):
        """将帧转换为base64"""
        # 缩放图片以减小传输大小
        small_frame = cv2.resize(frame, (640, 480))
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    def ask_vision_model_async(self, image_base64):
        """异步调用视觉模型"""
        try:
            messages = [HumanMessage(content=[
                {"type": "text", "text": self.question_template},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ])]
            response = self.vision_model.invoke(messages)
            self.ai_response_queue.put(response.content)
        except Exception as e:
            self.ai_response_queue.put(f"AI调用失败: {str(e)}")

    def process_frame(self, frame):
        """处理单帧：YOLO检测 + 添加AI描述文字"""
        # YOLO检测
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # 添加AI描述文字（放在顶部）
        h, w = annotated_frame.shape[:2]

        # 创建半透明背景条
        overlay = annotated_frame.copy()
        bar_height = 80
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)

        # 添加AI描述文字（自动换行）
        text = f"AI: {self.latest_ai_description}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)

        # 简单的文字换行处理
        max_chars_per_line = 50
        lines = []
        for i in range(0, len(text), max_chars_per_line):
            lines.append(text[i:i + max_chars_per_line])

        y_offset = 25
        for line in lines:
            cv2.putText(annotated_frame, line, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 20

        # 添加帧率和YOLO检测统计
        cv2.putText(annotated_frame, f"YOLO: {len(results[0].boxes)} objects detected",
                    (10, bar_height - 10), font, 0.5, (255, 255, 255), thickness)

        return annotated_frame

    def ai_worker(self):
        """AI处理的独立线程"""
        while self.running:
            try:
                # 获取待处理的帧
                frame_data = self.ai_response_queue.get(timeout=1)
                if frame_data is None:
                    break

                # 调用视觉模型
                img_base64 = frame_data["base64"]
                self.ask_vision_model_async(img_base64)

                # 处理AI响应
                if not self.ai_response_queue.empty():
                    response = self.ai_response_queue.get()
                    self.latest_ai_description = response[:200]  # 限制长度
                    print(f"[Frame {frame_data['frame_num']}] AI: {self.latest_ai_description}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AI worker error: {e}")

    def run(self):
        """主运行函数"""
        # 打开视频流
        cap = cv2.VideoCapture(self.rtsp_input)
        if not cap.isOpened():
            print(f"无法打开视频流：{self.rtsp_input}")
            return

        # 获取实际帧尺寸
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0:
            self.frame_width, self.frame_height = actual_width, actual_height

        # FFmpeg推流命令
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # 使用彩色格式
            '-s', f'{self.frame_width}x{self.frame_height}',
            '-r', '30',  # 帧率
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-f', 'rtsp',
            self.rtsp_output
        ]

        # 启动FFmpeg进程
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        # 启动AI工作线程
        ai_thread = threading.Thread(target=self.ai_worker)
        ai_thread.daemon = True
        ai_thread.start()

        print(f"开始处理视频流...")
        print(f"输入: {self.rtsp_input}")
        print(f"输出: {self.rtsp_output}")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧，退出")
                    break

                self.frame_count += 1

                # 处理帧（YOLO + 添加文字）
                processed_frame = self.process_frame(frame)

                # 每隔N帧调用一次AI
                if self.frame_count % self.ai_frame_interval == 0:
                    # 异步发送到AI处理队列
                    img_base64 = self.frame_to_base64(frame)
                    self.ai_response_queue.put({
                        "base64": img_base64,
                        "frame_num": self.frame_count
                    })

                # 推流到FFmpeg
                try:
                    process.stdin.write(processed_frame.tobytes())
                except (BrokenPipeError, OSError) as e:
                    print(f"FFmpeg进程错误: {e}")
                    break

                # 本地显示（可选）
                display_frame = cv2.resize(processed_frame, (960, 540))
                cv2.imshow("AI Video Processor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.cleanup(cap, process)

    def cleanup(self, cap, process):
        """清理资源"""
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        if process and process.poll() is None:
            process.stdin.close()
            process.wait()
        print("清理完成")


# 使用示例
if __name__ == "__main__":
    # RTSP地址配置
    INPUT_URL = "rtsp://127.0.0.1:25544/input"
    OUTPUT_URL = "rtsp://127.0.0.1:25544/output"

    processor = VideoAIProcessor(INPUT_URL, OUTPUT_URL)
    processor.run()