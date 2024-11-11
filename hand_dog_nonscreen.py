import time
import sys
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Value
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import mediapipe as mp_mediapipe
import signal


# 手势检测类：用于手腕翻转和特定手势的检测
class HandGestureDetector:
    def __init__(self, time_window=1.5):
        self.previous_flip_status = None  # 手掌向上 or 向下
        self.flip_count = 0  # 翻转计数
        self.hand_open_detected = False  # 标记是否检测到五指张开
        self.hand_open_detected_time = 0  # 记录五指张开的检测时间
        self.gesture_two_detected_time = 0  # 记录手势2的持续检测时间

    def detect_wrist_flip(self, landmarks):
        wrist_base = landmarks[0]
        thumb_tip = landmarks[4]
        return thumb_tip.y > wrist_base.y

    def process_flip(self, landmarks):
        current_flip_status = self.detect_wrist_flip(landmarks)
        if self.previous_flip_status is not None and self.previous_flip_status != current_flip_status:
            self.flip_count += 1
            if self.flip_count == 3:  # 检测到手腕翻转
                self.flip_count = 0
                return True  # 翻转完成
        self.previous_flip_status = current_flip_status
        return False

    def is_gesture_one(self, landmarks):
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        wrist = landmarks[0]
        return index_tip.y < wrist.y and middle_tip.y > wrist.y

    def is_gesture_two(self, landmarks):
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        return (index_tip.y < wrist.y and middle_tip.y < wrist.y and
                ring_tip.y > wrist.y and pinky_tip.y > wrist.y)

    def is_hand_open(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        return (thumb_tip.y < wrist.y and index_tip.y < wrist.y and
                middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y)

    def reset(self):
        self.previous_flip_status = None
        self.flip_count = 0
        self.hand_open_detected = False  # 重置五指张开状态
        self.hand_open_detected_time = 0  # 重置五指张开计时器
        self.gesture_two_detected_time = 0  # 重置手势2计时器


# 机器人运动控制类
class SportModeTest:
    def __init__(self) -> None:
        try:
            self.client = SportClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
        except AttributeError as e:
            print(f"Error initializing SportClient: {e}")

    def RiseSit(self):
        try:
            self.client.RiseSit()
            print("Stand up !!!")
        except Exception as e:
            print(f"Error executing RiseSit: {e}")

    def Sit(self):
        try:
            self.client.Sit()
            print("Sit down !!!")
        except Exception as e:
            print(f"Error executing Sit: {e}")

    def Hello(self):
        try:
            self.client.Hello()
            print("Hello !!!")
        except Exception as e:
            print(f"Error executing Hello: {e}")

    def Heart(self):
        try:
            self.client.Heart()
            print("Heart !!!")
        except Exception as e:
            print(f"Error executing Heart: {e}")


# 初始化视频客户端
def initialize_video_client(network_interface):
    ChannelFactoryInitialize(0, network_interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


# 获取视频帧
def get_video_frame(client):
    code, data = client.GetImageSample()
    if code == 0:
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image
    else:
        print("Get image sample error. code:", code)
        return None


# 绘制检测区域
def draw_detection_area(image, radius):
    height, width, _ = image.shape
    center = (width // 2, (height // 2) - 200)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


# 手势检测线程
def gesture_detection_process(pipe, network_interface, last_action_time_shared):
    # 在子进程中重新初始化
    client = initialize_video_client(network_interface)
    mp_hands = mp_mediapipe.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    gesture_detector = HandGestureDetector()
    radius = 800

    gesture_last_sent = None  # 保存上一次发送的手势指令

    while True:
        # 获取当前时间
        current_time = time.time()

        image = get_video_frame(client)
        if image is None:
            break

        mask = draw_detection_area(image, radius)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_image = cv2.bitwise_and(imgRGB, imgRGB, mask=mask)
        result = hands.process(masked_image)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                print("Detected hand landmarks")

                # 检测五指张开
                if not gesture_detector.hand_open_detected:
                    if gesture_detector.is_hand_open(handLms.landmark):
                        gesture_detector.hand_open_detected = True
                        gesture_detector.hand_open_detected_time = current_time  # 记录五指张开时间
                        print("Hand open detected, entering gesture detection mode")
                else:
                    if current_time - gesture_detector.hand_open_detected_time > 5:
                        gesture_detector.reset()
                        print("No actions detected within 5 seconds, resetting to hand open detection")
                        continue

                    if gesture_detector.process_flip(handLms.landmark) and gesture_last_sent != "flip_detected":
                        print("Flip detected")
                        pipe.send("flip_detected")

                        gesture_detector.reset()
                        gesture_last_sent = "flip_detected"
                    elif gesture_detector.is_gesture_one(handLms.landmark) and gesture_last_sent != "gesture_one_detected":
                        print("Gesture 1 detected")
                        pipe.send("gesture_one_detected")

                        gesture_detector.reset()
                        gesture_last_sent = "gesture_one_detected"
                    elif gesture_detector.is_gesture_two(handLms.landmark) and gesture_last_sent != "gesture_two_detected":
                        print("Gesture 2 detected")
                        pipe.send("gesture_two_detected")

                        gesture_detector.reset()
                        gesture_last_sent = "gesture_two_detected"
        else:
            gesture_last_sent = None  # 如果没有手势，重置最后发送的手势


# 机器人运动控制线程
def robot_control_process(pipe, network_interface, last_action_time_shared):
    # 在子进程中重新初始化
    ChannelFactoryInitialize(0, network_interface)
    robot = SportModeTest()
    dog_is_standing = False

    while True:
        try:
            gesture = pipe.recv()  # 等待接收手势指令
            print(f"Received gesture: {gesture}")

            if gesture == "flip_detected":
                if dog_is_standing:
                    robot.Sit()
                    dog_is_standing = False
                else:
                    robot.RiseSit()
                    dog_is_standing = True

            elif gesture == "gesture_one_detected":
                robot.Hello()

            elif gesture == "gesture_two_detected":
                robot.Heart()

        except EOFError:
            break


# 捕获信号处理，确保子进程能够优雅关闭
def signal_handler(sig, frame):
    print("Signal received, terminating...")
    sys.exit(0)


# 主函数
def main():
    # 捕获系统信号 (如 Ctrl+C) 并终止进程
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parent_conn, child_conn = mp.Pipe()
    network_interface = sys.argv[1]  # 获取网络接口

    # 创建一个共享的冷却时间变量
    last_action_time_shared = Value('d', time.time())

    # 创建手势检测进程和机器人控制进程
    p1 = mp.Process(target=gesture_detection_process, args=(parent_conn, network_interface, last_action_time_shared))
    p2 = mp.Process(target=robot_control_process, args=(child_conn, network_interface, last_action_time_shared))

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, terminating processes...")
    finally:
        # 确保所有子进程被终止
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()


if __name__ == "__main__":
    main()
