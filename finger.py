import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import math


class HandGestureVolumeControl:
    def __init__(self):
        # 保持原有MediaPipe参数
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 初始化音量控制
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.volume_range = self.volume.GetVolumeRange()
        self.current_volume = self.volume.GetMasterVolumeLevel()
        self.is_muted = self.volume.GetMute()

        # 静音功能参数
        self.last_volume_before_mute = self.current_volume
        self.mute_cooldown = 0.8
        self.last_mute_time = 0
        self.two_finger_counter = 0
        self.required_two_finger_frames = 3

        # 新增：防止误判的激活/暂停机制
        self.volume_active = False  # 实际控制音量的开关
        self.activation_gesture_detected = False  # 激活手势检测标记
        self.activation_counter = 0
        self.required_activation_frames = 5  # 需要连续5帧检测到激活手势
        self.pause_gesture_detected = False  # 暂停手势检测标记
        self.pause_counter = 0
        self.required_pause_frames = 5

        # 原有手势控制参数
        self.wakeup_duration = 2
        self.volume_sensitivity = 0.8
        self.cooldown = 0.1

        # 状态变量
        self.volume_mode = False
        self.fist_start_time = None
        self.wakeup_progress = 0
        self.last_action_time = 0

        # 倒计时和调试变量
        self.countdown_start_time = None
        self.countdown_remaining = 0
        self.fist_detection_count = 0
        self.required_fist_frames = 10

        # 手指状态跟踪
        self.prev_finger_states = None
        self.change_counter = [0] * 5
        self.current_finger_count = 0
        self.debug_info = ""

    def count_fingers(self, landmarks):
        """保持原有手指计数算法"""
        if not landmarks:
            self.current_finger_count = 0
            self.debug_info = "No hand detected"
            return 0, False

        finger_tips = [4, 8, 12, 16, 20]
        palm_center = landmarks[0]
        finger_states = [0] * 5
        self.debug_info = ""

        # 计算手掌中心和大小
        palm_points = [0, 5, 9, 13, 17]
        palm_points_coords = [landmarks[i] for i in palm_points]
        palm_center = np.mean(palm_points_coords, axis=0).astype(int)
        palm_radius = max(math.hypot(p[0] - palm_center[0], p[1] - palm_center[1])
                          for p in palm_points_coords)

        # 识别拇指
        thumb_tip = landmarks[4]
        dist_thumb_palm = math.hypot(thumb_tip[0] - palm_center[0], thumb_tip[1] - palm_center[1])
        is_thumb_close = dist_thumb_palm < palm_radius * 1.2
        finger_states[0] = 0 if is_thumb_close else 1

        # 其他手指检测
        for i in range(1, 5):
            tip_idx = finger_tips[i]
            tip = landmarks[tip_idx]
            dist_tip_palm = math.hypot(tip[0] - palm_center[0], tip[1] - palm_center[1])
            is_finger_close = dist_tip_palm < palm_radius * 1.3
            finger_states[i] = 0 if is_finger_close else 1

        # 平滑处理
        if self.prev_finger_states is not None:
            for i in range(5):
                if finger_states[i] != self.prev_finger_states[i]:
                    self.change_counter[i] += 1
                    if self.change_counter[i] < 1:
                        finger_states[i] = self.prev_finger_states[i]
                    else:
                        self.change_counter[i] = 0
                else:
                    self.change_counter[i] = 0

        self.prev_finger_states = finger_states.copy()
        total_fingers = sum(finger_states)
        is_fist = total_fingers <= 1

        # 调试信息
        fingers_status = [f"F{i}:{s}" for i, s in enumerate(finger_states)]
        self.debug_info = (f"Fingers: {total_fingers}, " + ", ".join(fingers_status) +
                           f", Palm radius: {int(palm_radius)}")

        self.current_finger_count = total_fingers
        return total_fingers, is_fist

    def check_activation_gestures(self, finger_count):
        """检测激活和暂停手势，防止误判"""
        # 激活手势：特定的4指手势（拇指收起，其他四指伸出）
        # 这样普通的手部出现不会触发，需要刻意做这个手势
        if finger_count == 4:
            self.activation_counter += 1
            self.pause_counter = 0  # 重置暂停计数器
            self.debug_info += f", Activate frames: {self.activation_counter}"

            if self.activation_counter >= self.required_activation_frames and not self.volume_active:
                self.volume_active = True
                self.activation_gesture_detected = True
                print("Volume control ACTIVATED (4 fingers gesture)")
                return
        else:
            self.activation_counter = 0

        # 暂停手势：特定的1指手势（仅伸出食指）
        if finger_count == 1:
            # 检查是否是食指伸出（F1=1，其他为0）
            if self.prev_finger_states and self.prev_finger_states[1] == 1 and sum(self.prev_finger_states) == 1:
                self.pause_counter += 1
                self.activation_counter = 0  # 重置激活计数器
                self.debug_info += f", Pause frames: {self.pause_counter}"

                if self.pause_counter >= self.required_pause_frames and self.volume_active:
                    self.volume_active = False
                    self.pause_gesture_detected = True
                    print("Volume control PAUSED (1 finger gesture)")
                    return
        else:
            self.pause_counter = 0

    def check_wakeup(self, is_fist):
        """保持原有唤醒检测逻辑"""
        current_time = time.time()

        if is_fist and self.current_finger_count <= 1:
            self.fist_detection_count += 1
            self.debug_info += f", Fist frames: {self.fist_detection_count}"

            if self.fist_detection_count >= self.required_fist_frames:
                if self.fist_start_time is None:
                    self.fist_start_time = current_time
                    self.countdown_start_time = current_time
                else:
                    elapsed = current_time - self.fist_start_time
                    self.wakeup_progress = min(100, (elapsed / self.wakeup_duration) * 100)
                    self.countdown_remaining = max(0, self.wakeup_duration - elapsed)

                    if elapsed >= self.wakeup_duration:
                        self.volume_mode = True
                        self.fist_start_time = None
                        self.wakeup_progress = 0
                        self.countdown_start_time = None
                        self.countdown_remaining = 0
                        print("Volume mode activated!")
        else:
            self.fist_detection_count = 0
            self.fist_start_time = None
            self.wakeup_progress = 0
            self.countdown_start_time = None
            self.countdown_remaining = 0

    def adjust_volume(self, finger_count, is_fist):
        """只有在激活状态下才调节音量，防止误判"""
        if not self.volume_mode:
            return

        # 先检测激活和暂停手势
        self.check_activation_gestures(finger_count)

        # 如果没有激活，不执行任何音量操作
        if not self.volume_active:
            self.debug_info += ", Control: PAUSED"
            return
        else:
            self.debug_info += ", Control: ACTIVE"

        current_time = time.time()

        # 二指静音功能
        if finger_count == 2:
            self.two_finger_counter += 1
            self.debug_info += f", 2-finger frames: {self.two_finger_counter}"

            if (current_time - self.last_mute_time > self.mute_cooldown and
                    self.two_finger_counter >= self.required_two_finger_frames):
                if not self.is_muted:
                    self.last_volume_before_mute = self.current_volume
                    self.volume.SetMute(True, None)
                    self.is_muted = True
                    print("Muted successfully (2 fingers)")
                    self.last_mute_time = current_time
                return
        else:
            self.two_finger_counter = 0

        # 三指解除静音功能
        if finger_count == 3:
            if self.is_muted:
                self.volume.SetMute(False, None)
                self.current_volume = self.last_volume_before_mute
                self.volume.SetMasterVolumeLevel(self.current_volume, None)
                self.is_muted = False
                print("Unmuted successfully (3 fingers)")
                self.last_mute_time = current_time
                return

        # 静音状态下不响应其他音量调节
        if self.is_muted:
            return

        if current_time - self.last_action_time < self.cooldown:
            return

        min_vol, max_vol, _ = self.volume_range
        step = (max_vol - min_vol) * self.volume_sensitivity / 100

        # 3-5指增大音量
        if finger_count >= 3:
            self.current_volume = min(self.current_volume + step, max_vol)
            self.volume.SetMasterVolumeLevel(self.current_volume, None)
            self.last_action_time = current_time

        # 0-1指减小音量
        elif is_fist or finger_count <= 1:
            self.current_volume = max(self.current_volume - step, min_vol)
            self.volume.SetMasterVolumeLevel(self.current_volume, None)
            self.last_action_time = current_time

    def draw_info(self, frame, finger_count, is_fist):
        """显示激活/暂停状态，帮助用户了解当前控制状态，新增音量条显示"""
        height, width = frame.shape[:2]
        volume_percent = int((self.current_volume - self.volume_range[0]) /
                             (self.volume_range[1] - self.volume_range[0]) * 100)

        # 显示音量和静音状态
        if self.is_muted:
            cv2.putText(frame, "MUTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Volume: {volume_percent}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 新增：绘制音量条
        vol_bar_x, vol_bar_y = 10, 50
        vol_bar_width, vol_bar_height = 300, 20
        # 音量条背景
        cv2.rectangle(frame, (vol_bar_x, vol_bar_y),
                      (vol_bar_x + vol_bar_width, vol_bar_y + vol_bar_height),
                      (50, 50, 50), -1)
        # 音量条前景
        vol_level = int(vol_bar_width * (volume_percent / 100))
        # 根据音量大小设置颜色（低音量蓝色，中音量绿色，高音量红色）
        if volume_percent < 30:
            bar_color = (255, 0, 0)  # 蓝色
        elif volume_percent < 70:
            bar_color = (0, 255, 0)  # 绿色
        else:
            bar_color = (0, 0, 255)  # 红色
        cv2.rectangle(frame, (vol_bar_x, vol_bar_y),
                      (vol_bar_x + vol_level, vol_bar_y + vol_bar_height),
                      bar_color, -1)
        # 音量条边框
        cv2.rectangle(frame, (vol_bar_x, vol_bar_y),
                      (vol_bar_x + vol_bar_width, vol_bar_y + vol_bar_height),
                      (255, 255, 255), 1)

        # 显示当前检测到的手指数量
        finger_text = f"Fingers: {finger_count}"
        cv2.putText(frame, finger_text, (width - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # 显示控制激活状态
        control_status = "ACTIVE" if self.volume_active else "PAUSED"
        status_color = (0, 255, 0) if self.volume_active else (0, 165, 255)  # 绿色=激活，橙色=暂停
        cv2.putText(frame, f"Control: {control_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 显示模式状态
        mode_text = "Volume Mode (Active)" if self.volume_mode else "Standby Mode"
        mode_color = (0, 0, 255) if self.volume_mode else (0, 255, 0)
        cv2.putText(frame, mode_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # 显示唤醒进度条
        if not self.volume_mode and self.countdown_start_time is not None:
            bar_x, bar_y = 10, 150
            bar_width, bar_height = 300, 30
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (50, 50, 50), -1)
            progress = int(bar_width * (1 - self.countdown_remaining / self.wakeup_duration))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height),
                          (0, 255, 0), -1)
            cv2.putText(frame, f"Activating in: {int(self.countdown_remaining)}s",
                        (bar_x + bar_width + 10, bar_y + bar_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示操作提示
        if self.volume_mode:
            # 激活/暂停手势提示
            cv2.putText(frame, "4 fingers: Activate control", (10, height - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # 黄色提示
            cv2.putText(frame, "1 finger (index): Pause control", (10, height - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 原有操作提示
            cv2.putText(frame, "Open palm (3-5 fingers): Increase volume", (10, height - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Make a fist (0-1 fingers): Decrease volume", (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "2 fingers: Mute", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "3 fingers: Unmute", (width - 250, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Make a fist (0-1 fingers) to activate", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示调试信息
        cv2.putText(frame, self.debug_info, (10, height - 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return frame

    def run(self):
        """保持原有主循环逻辑"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Hand Gesture Volume Control Started")
        print("1. Make a fist (0-1 fingers) for 2 seconds to enter volume mode")
        print("2. In volume mode:")
        print("   - 4 fingers: Activate volume control")
        print("   - 1 finger (index only): Pause volume control")
        print("   - 3-5 fingers: Increase volume")
        print("   - 0-1 fingers: Decrease volume")
        print("   - 2 fingers: Mute")
        print("   - 3 fingers: Unmute")
        print("3. Press 'r' to exit mode, 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read camera feed")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            finger_count = 0
            is_fist = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w = frame.shape[:2]
                    landmarks = [(int(lm.x * w), int(lm.y * h))
                                 for lm in hand_landmarks.landmark]

                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                    finger_count, is_fist = self.count_fingers(landmarks)

                    if not self.volume_mode:
                        self.check_wakeup(is_fist)
                    else:
                        self.adjust_volume(finger_count, is_fist)

            frame = self.draw_info(frame, finger_count, is_fist)
            cv2.imshow('Hand Gesture Volume Control', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.volume_mode = False
                self.volume_active = False  # 退出模式时同时暂停控制
                print("Exited volume mode")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        controller = HandGestureVolumeControl()
        controller.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")