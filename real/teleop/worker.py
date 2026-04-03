import json
import lzma
import os
import pickle
import queue
import sys
import threading
import time
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from teleop.utils.logger import logger
from vr import VuerTeleop
from vr_pico import PicoTeleop
from writers import AsyncImageWriter, AsyncWriter

import socket
import struct
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# from turbojpeg import TJPF_BGR, TurboJPEG


FREQ = 30
DELAY = 1 / FREQ

from teleop.constants import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class PicoIRStreamer:
    def __init__(self, pico_ip, port, width=1280, height=720, fps=30):
        self.pico_ip = pico_ip
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps

        self._running = False
        self._connected = False
        self._sock = None
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0

        self._latest_frame = None
        self._lock = threading.Lock()

    def start(self):
        Gst.init(None)
        self._running = True
        self._start_pipeline()
        threading.Thread(target=self._connection_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()
        print(f"[PicoIRStreamer] started, target={self.pico_ip}:{self.port}")

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._latest_frame = frame_bgr.copy()

    def _connection_loop(self):
        while self._running:
            if not self._connected:
                old = self._sock
                self._sock = None
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass

                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect((self.pico_ip, self.port))
                    s.settimeout(None)
                    self._sock = s
                    self._connected = True
                    print(f"[PicoIRStreamer] connected to {self.pico_ip}:{self.port}")
                except Exception:
                    pass

            time.sleep(1.0)

    def _start_pipeline(self):
        pipe_str = (
            f"appsrc name=src is-live=True format=time ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 ! "
            f"video/x-h264,profile=baseline ! "
            f"h264parse config-interval=-1 ! "
            f"video/x-h264,stream-format=byte-stream,alignment=au ! "
            f"appsink name=sink emit-signals=True sync=False"
        )
        self._pipeline = Gst.parse_launch(pipe_str)
        self._appsrc = self._pipeline.get_by_name("src")
        appsink = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(Gst.State.PLAYING)

    def _on_encoded_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            if self._connected and self._sock is not None:
                try:
                    header = struct.pack(">I", len(info.data))
                    self._sock.sendall(header + info.data)
                except Exception:
                    self._connected = False
                    print("[PicoIRStreamer] connection lost, retrying...")
            buf.unmap(info)
        return Gst.FlowReturn.OK

    def _push_loop(self):
        dt = 1.0 / self.fps
        while self._running:
            t0 = time.time()

            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame = None 

            if frame is not None and self._appsrc is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                gst_buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(frame).tobytes())
                gst_buf.pts = self._frame_id * (Gst.SECOND // self.fps)
                gst_buf.duration = Gst.SECOND // self.fps
                self._appsrc.emit("push-buffer", gst_buf)
                self._frame_id += 1

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def stop(self):
        self._running = False
        self._connected = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None



class TeleoperatorProcess:
    def __init__(self, teleop_shm_array, img_shm_name, kill_event, ir_data_queue, session_start_event, is_pico_streamer=False):
        self.teleop_shm_array = teleop_shm_array
        self.kill_event = kill_event
        self.ir_data_queue = ir_data_queue
        self.session_start_event = session_start_event
        self.is_pico_streamer = is_pico_streamer

        # Connect to the shared memory for images created by the worker
        self.img_shm = shared_memory.SharedMemory(name=img_shm_name)
        height, width = 720, 1280  # Match the dimensions expected by teleoperator
        self.img_array = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=self.img_shm.buf
        )

        if self.is_pico_streamer:
            self.teleoperator = PicoTeleop()
        else:
            # Pass the shared memory name so that VuerTeleop attaches to the same memory
            self.teleoperator = VuerTeleop(img_shm_name)

    def _ir_loop(self):
        """Thread that processes IR image data and copies it to shared memory."""
        while not self.kill_event.is_set():
            try:
                raw_ir_data = self.ir_data_queue.get(timeout=0.01)
                raw_ir_np = np.frombuffer(raw_ir_data, dtype=np.uint8)
                decoded_frame = cv2.imdecode(raw_ir_np, cv2.IMREAD_COLOR)
                if decoded_frame is not None:
                    resized_frame = cv2.resize(
                        decoded_frame, (1280, 720), interpolation=cv2.INTER_LINEAR
                    )
                    np.copyto(self.img_array, resized_frame)
                else:
                    logger.warning("Teleoperator: Failed to decode IR frame.")
            except queue.Empty:
                pass
            # A small sleep can help reduce CPU usage
            time.sleep(0.005)

    def _step_loop(self):
        """Thread that repeatedly calls teleoperator.step() and updates shared state."""
        prev_session_active = self.session_start_event.is_set()
        while not self.kill_event.is_set():
            curr_session_active = self.session_start_event.is_set()
            if curr_session_active and not prev_session_active:
                self.teleoperator.processor.trigger_calibration()
            prev_session_active = curr_session_active

            head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                self.teleoperator.step(full_head=True)
            )
            # logger.debug("Teleoperator: finished stepping")
            self.teleop_shm_array[0:16] = head_rmat.flatten()
            self.teleop_shm_array[16:32] = left_pose.flatten()
            self.teleop_shm_array[32:48] = right_pose.flatten()
            self.teleop_shm_array[48:55] = np.array(left_qpos).flatten()
            self.teleop_shm_array[55:62] = np.array(right_qpos).flatten()
            time.sleep(0.01)

    def run(self):
        # logger.info(f"Teleoperator process started with PID {os.getpid()}")
        # Start IR processing and teleoperator stepping in separate threads
        ir_thread = threading.Thread(target=self._ir_loop, daemon=True)
        step_thread = threading.Thread(target=self._step_loop, daemon=True)
        ir_thread.start()
        step_thread.start()

        try:
            while not self.kill_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Teleoperator process encountered an error: {e}")
        finally:
            logger.info("Teleoperator process shutting down")
            self.kill_event.set()
            ir_thread.join()
            step_thread.join()
            self.teleoperator.shutdown()
            self.img_shm.close()
            logger.info("Teleoperator process exited")


class RobotDataWorker:
    def __init__(self, shared_data, robot_shm_array, teleop_shm_array, robot="h1", is_pico_streamer=False, pico_ip="192.168.0.128"):
        self.robot = robot
        self.shared_data = shared_data
        self.kill_event = shared_data["kill_event"]
        self.session_start_event = shared_data["session_start_event"]
        self.end_event = shared_data["end_event"]  # TODO: redundent
        # self.h1_lock = Lock()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if robot == "h1":
            self.socket.connect("tcp://192.168.123.162:5556")
        else:
            self.socket.connect("tcp://192.168.123.164:5556")

        self.robot_shm_array = robot_shm_array
        self.teleop_shm_array = teleop_shm_array
        self.depth_kill_event = Event()
        self.teleop_kill_event = Event()

        self.is_pico_streamer = is_pico_streamer
        self.pico_ip = pico_ip

        height, width = 720, 1280  # Match the dimensions expected by teleoperator
        self.img_shm = shared_memory.SharedMemory(
            create=True, size=height * width * 3 * np.uint8().itemsize
        )
        self.img_array = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=self.img_shm.buf
        )

        self.depth_queue = Queue()
        self.async_image_writer = AsyncImageWriter()

        self.depth_proc = Process(
            target=self.depth_writer_process,
            args=(self.depth_queue, self.depth_kill_event),
        )

        self.ir_data_queue = Queue()
        self.teleop_proc = Process(
            target=self._run_teleoperator_process,
            args=(
                self.teleop_shm_array,
                self.img_shm.name,
                self.teleop_kill_event,
                self.ir_data_queue,
                self.session_start_event,
            ),
        )
        self.teleop_proc.start()
        logger.debug(
            f"Worker started teleoperator process with PID {self.teleop_proc.pid}"
        )

        if self.is_pico_streamer:

            self.pico_streamer = PicoIRStreamer(
                pico_ip=self.pico_ip,
                port=12345,
                width=1280,
                height=720,
                fps=30,
            )
            self.pico_streamer.start()

        # resetable vars
        self.frame_idx = 0
        self.last_robot_data = None
        self.robot_data_writer = None

    def _run_teleoperator_process(
        self, teleop_shm_array, img_shm_name, kill_event, ir_data_queue, session_start_event
    ):
        teleop_process = TeleoperatorProcess(
            teleop_shm_array, img_shm_name, kill_event, ir_data_queue, session_start_event, self.is_pico_streamer
        )
        teleop_process.run()

    def depth_writer_process(self, depth_queue, kill_event):
        while not kill_event.is_set():
            try:
                filename, depth_array = depth_queue.get(timeout=0.5)
                # buffer = io.BytesIO()
                # np.save(buffer, depth_array)
                # depth_bytes = buffer.getvalue()
                compressed_data = lzma.compress(depth_array.tobytes(), preset=0)
                with open(filename, "wb") as f:
                    f.write(compressed_data)
            except queue.Empty:
                continue

    def dump_state(self, filename=None):
        """Dump current system state for debugging"""
        if filename is None:
            filename = f"debug_dump_{time.strftime('%Y%m%d_%H%M%S')}.pkl"

        state = {
            "h1_data": self.robot_shm_array.copy(),
            "teleop_data": self.teleop_shm_array.copy(),
            "frame_idx": self.frame_idx if hasattr(self, "frame_idx") else None,
            "timestamp": time.time(),
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"State dumped to {filename}")

    def _sleep_until_mod33(self, time_curr):
        integer_part = int(time_curr)
        decimal_part = time_curr - integer_part
        ms_part = int(decimal_part * 1000) % 100

        next_ms_part = ((ms_part // 33) + 1) * 33 % 100
        hundred_ms_part = int(decimal_part * 10 % 10)
        if next_ms_part == 32:
            hundred_ms_part += 1

        next_capture_time = integer_part + next_ms_part / 1000 + hundred_ms_part / 10
        if (next_capture_time - time_curr) < 0:
            next_capture_time += 1
        time.sleep(next_capture_time - time_curr)

    def _recv_zmq_frame(self) -> Tuple[Any, Any, Any]:
        logger.debug("worker: start sending request")
        self.socket.send(b"get_frame")

        rgb_bytes, ir_bytes, depth_bytes = (
            self.socket.recv_multipart()
        )  # resource leak due to blocking?
        logger.debug("worker: recv request")

        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        ir_array = np.frombuffer(ir_bytes, np.uint8)
        depth_array = np.frombuffer(depth_bytes, np.uint16).reshape((480, 640))
        logger.debug("worker: processed depth array")

        return rgb_array, ir_array, depth_array

    def extract_usable(self, row):
        """
        Extract usable pressure readings from a row using fixed index patterns.

        Pattern:
          - Type A row: if the first element is valid (not 0 or 30000), use indices [0, 2, 9, 11]
          - Type B row: if the first element is waste but index 3 is valid, use indices [3, 6, 8]
          - Otherwise, return None (i.e. the row contains no usable data)
        """
        # Check if the row is completely waste
        if all(val in (0.0, 30000.0) for val in row):
            return None, None

        # Type A: valid if the first element is not a waste value
        if row[0] not in (0.0, 30000.0):
            usable = [row[i] for i in [0, 2, 9, 11]]
            sensor_type = "A"
        # Type B: use alternative fixed indices if index 3 is valid
        elif row[3] not in (0.0, 30000.0):
            usable = [row[i] for i in [3, 6, 8]]
            sensor_type = "B"
        else:
            # No valid data found based on fixed positions
            return None, None

        return sensor_type, usable

    def format_pressure_data(self, data):
        sensors = []
        for idx, row in enumerate(data, start=1):
            sensor_type, readings = self.extract_usable(row)
            if readings is not None:
                sensors.append(
                    {
                        "sensor_id": idx,
                        "sensor_type": sensor_type,
                        "usable_readings": readings,
                    }
                )
        return sensors

    def get_robot_data(self, time_curr):
        logger.debug(f"worker: starting to get robot data")
        # with self.h1_lock:
        robot_data = self.robot_shm_array.copy()

        # Define starting indices for each data section
        if self.robot == "h1":
            robot_sizes = H1_sizes
        elif self.robot == "g1":
            robot_sizes = G1_sizes

        leg_start = 0
        arm_start = robot_sizes.LEG_STATE_SIZE
        hand_start = arm_start + robot_sizes.ARM_STATE_SIZE
        imu_start = hand_start + robot_sizes.HAND_STATE_SIZE

        # Extract individual components
        legstate = robot_data[leg_start:arm_start]
        armstate = robot_data[arm_start:hand_start]
        handstate = robot_data[hand_start:imu_start]

        # Extract IMU data
        imu_accelerometer_start = imu_start + robot_sizes.IMU_QUATERNION_SIZE
        imu_gyroscope_start = (
            imu_accelerometer_start + robot_sizes.IMU_ACCELEROMETER_SIZE
        )
        imu_rpy_start = imu_gyroscope_start + robot_sizes.IMU_GYROSCOPE_SIZE
        imu_rpy_end = imu_rpy_start + robot_sizes.IMU_RPY_SIZE


        imustate = {
            "quaternion": robot_data[imu_start:imu_accelerometer_start].tolist(),
            "accelerometer": robot_data[
                imu_accelerometer_start:imu_gyroscope_start
            ].tolist(),
            "gyroscope": robot_data[imu_gyroscope_start:imu_rpy_start].tolist(),
            "rpy": robot_data[imu_rpy_start:imu_rpy_end].tolist(),
        }
        
        # Extract ODOM data
        odom_velocity_start = imu_rpy_end + robot_sizes.ODOM_POSITION_SIZE
        odom_rpy_start = odom_velocity_start + robot_sizes.ODOM_VELOCITY_SIZE
        odom_quat_start = odom_rpy_start + robot_sizes.ODOM_RPY_SIZE
        odom_quat_end = odom_quat_start + robot_sizes.ODOM_QUATERNION_SIZE

        odomstate = {
            "position": robot_data[imu_rpy_end:odom_velocity_start].tolist(),
            "velocity": robot_data[odom_velocity_start:odom_rpy_start].tolist(),
            "rpy": robot_data[odom_rpy_start:odom_quat_start].tolist(),
            "quat": robot_data[odom_quat_start:odom_quat_end].tolist(),
        }

        pressure_state = None
        if self.robot == "g1":
            pressure_state = robot_data[
                odom_quat_end : odom_quat_end + robot_sizes.HAND_PRESS_SIZE
            ]

        robot_data = {
            "time": time_curr,
            "robot_type": self.robot,
            "states": {
                "arm_state": armstate.tolist(),
                "leg_state": legstate.tolist(),
                "hand_state": handstate.tolist(),
                "hand_pressure_state": (
                    self.format_pressure_data(pressure_state.reshape(18, 12).tolist())
                    if pressure_state is not None
                    else None
                ),
                "imu": imustate,
                "odometry": odomstate,
            },
            "actions": None,
            "image": f"color/frame_{self.frame_idx:06d}.jpg",
            "depth": f"depth/frame_{self.frame_idx:06d}.npy.lzma",
        }
        # logger.debug(f"worker: finish getting robot data")
        return robot_data

    def start(self):
        # logger.debug(f"Worker: Process ID (PID) {os.getpid()}")
        # self.teleop_proc.start()
        self.depth_proc.start()
        try:
            while not self.end_event.is_set():
                logger.info(
                    "Worker: waiting for new session start (session_start_event)."
                )
                self.session_start_event.wait()
                logger.info("Worker: starting new session.")
                self.run_session()
                self.async_image_writer.close()
                self.async_image_writer = AsyncImageWriter()
        finally:
            self.socket.close()
            self.context.term()

            self.teleop_kill_event.set()
            self.teleop_proc.join()
            logger.info("Worker: teleoperator process joined.")
            self.img_shm.close()
            self.img_shm.unlink()

            if hasattr(self, "pico_streamer") and self.pico_streamer is not None:
                self.pico_streamer.stop()

            # self.teleoperator.shutdown()
            self.depth_kill_event.set()
            self.depth_proc.join()
            logger.info("Worker: exited")

    def _write_image_data(self, rgb_array, depth_array):
        logger.debug("Worker: writing robot data")

        color_filename = os.path.join(
            self.shared_data["dirname"], f"color/frame_{self.frame_idx:06d}.jpg"
        )
        depth_filename = os.path.join(
            self.shared_data["dirname"], f"depth/frame_{self.frame_idx:06d}.npy.lzma"
        )

        if rgb_array is not None and depth_array is not None:
            self.async_image_writer.write_image(color_filename, rgb_array)
            self.depth_queue.put((depth_filename, depth_array))
            logger.debug(
                f"Saved color frame to {color_filename} and depth frame to {depth_filename}"
            )
        else:
            logger.error(f"failed to save image {self.frame_idx}")

    def _write_robot_data(self, rgb_array, depth_array, reuse=False):
        logger.debug(f"Worker: writing robot data")
        self._write_image_data(rgb_array, depth_array)

        robot_data = self.get_robot_data(time.time())

        if reuse:
            self.last_robot_data["time"] = time.time()
            self.robot_data_writer.write(json.dumps(robot_data))
        else:
            if self.robot_data_writer is not None:
                self.robot_data_writer.write(json.dumps(robot_data))
        self.last_robot_data = robot_data
        self.frame_idx += 1

    def _send_image_to_teleoperator(self, ir_array):
        self.ir_data_queue.put(ir_array)
    
    
    def _send_ir_to_pico(self, ir_array):
        if ir_array is None:
            return
        try:
            ir_np = np.frombuffer(ir_array, dtype=np.uint8)
            ir_img = cv2.imdecode(ir_np, cv2.IMREAD_COLOR)

            if ir_img is None:
                logger.warning("Failed to decode IR image")
                return

            ir_img = cv2.resize(ir_img, (1280, 720), interpolation=cv2.INTER_LINEAR)

            if hasattr(self, "pico_streamer") and self.pico_streamer is not None:
                self.pico_streamer.submit_frame(ir_img)

        except Exception as e:
            logger.error(f"Failed to send IR to Pico: {e}")


    def _session_init(self):
        if "dirname" not in self.shared_data:
            logger.error("Worker: failed to get dirname")
            exit(-1)
        self.robot_data_writer = AsyncWriter(
            os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
        )

    def process_data(self):
        logger.debug("request frame")
        rgb_array, ir_array, depth_array = self._recv_zmq_frame()
        logger.debug("got frame")
        if self.is_pico_streamer:
            self._send_ir_to_pico(ir_array)
        else:
            self._send_image_to_teleoperator(ir_array)
        time_curr = time.time()

        # logger.debug(f"Worker: got image")
        if self.is_first:
            self.is_first = False
            time.sleep(0.2)  # wait for master to populate data
            self._sleep_until_mod33(time.time())
            self.initial_capture_time = time.time()
            self._write_robot_data(rgb_array, depth_array)
            logger.debug(f"Worker: initial_capture_time is {self.initial_capture_time}")
            return

        next_capture_time = self.initial_capture_time + self.frame_idx * DELAY
        time_curr = time.time()
        logger.debug(
            f"[worker process] next_capture_time - time_curr: {next_capture_time - time_curr}"
        )

        if next_capture_time - time_curr >= 0:
            time.sleep(next_capture_time - time_curr)
            self._write_robot_data(rgb_array, depth_array)
        else:
            logger.warning(
                "worker: runner did not finish within 33ms, reusing previous data"
            )
            if self.last_robot_data is not None:
                self._write_robot_data(rgb_array, depth_array, reuse=True)
            else:
                logger.error("worker: no previous data available, generating null data")
                self._write_robot_data(None, None, reuse=True)

    def run_session(self):
        self._session_init()
        self.is_first = True
        try:
            while not self.kill_event.is_set():
                logger.debug("Worker: entering main loop")
                self.process_data()
                # self.robot_data_writer.close()
                # self.robot_data_writer = AsyncWriter(
                #     os.path.join(self.shared_data["dirname"], "robot_data.jsonl")
                # )
                logger.debug("Worker: initing robot_data_writer")

        # except Exception as e:
        #     logger.error(f"robot_data_worker encountered an error: {e}")

        finally:
            logger.info("Worker begin exiting.")
            # TODO: flush the buffer?
            # self.teleop_thread.join(1)
            logger.info("Worker: teleop thread joined.")
            self.robot_data_writer.close()
            logger.info("Worker: writer closed.")
            self.reset()
            logger.info("Worker: closing async image writer.")
            if hasattr(self, "async_image_writer"):
                self.async_image_writer.close()
                logger.info("Worker: async image writer closed.")

            logger.info("Worker process has exited.")

    def reset(self):
        self.frame_idx = 0
        self.initial_capture_time = None
