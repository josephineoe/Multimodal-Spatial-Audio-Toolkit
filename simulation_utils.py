import csv
import os
import time
import math

class SimulationLogger:
    def __init__(self, imu_csv="imu_log.csv", vision_csv="vision_log.csv"):
        self.imu_csv = imu_csv
        self.vision_csv = vision_csv

        # Initialize IMU log
        if not os.path.exists(self.imu_csv):
            with open(self.imu_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "qw", "qx", "qy", "qz", "roll", "pitch", "yaw"])

        # Initialize vision log
        if not os.path.exists(self.vision_csv):
            with open(self.vision_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "azimuth_deg", "elevation_deg", "distance_m", "conf", "cls_name"])

    def log_imu(self, timestamp, qw, qx, qy, qz, roll, pitch, yaw):
        with open(self.imu_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, qw, qx, qy, qz, roll, pitch, yaw])

    def log_vision(self, timestamp, az_deg, el_deg, dist_m, conf, cls_name):
        with open(self.vision_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, az_deg, el_deg, dist_m, conf, cls_name])

class SimulationReader:
    def __init__(self, imu_csv="imu_log.csv", vision_csv="vision_log.csv"):
        self.imu_csv = imu_csv
        self.vision_csv = vision_csv
        self.imu_data = []
        self.vision_data = []
        self.imu_index = 0
        self.vision_index = 0

        # Load data if exists
        if os.path.exists(self.imu_csv):
            with open(self.imu_csv, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                self.imu_data = [row for row in reader]

        if os.path.exists(self.vision_csv):
            with open(self.vision_csv, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                self.vision_data = [row for row in reader]

    def get_next_imu(self):
        if self.imu_data:
            if self.imu_index >= len(self.imu_data):
                self.imu_index = 0  # loop
            row = self.imu_data[self.imu_index]
            self.imu_index += 1
            return [float(x) for x in row[1:]]  # qw,qx,qy,qz,roll,pitch,yaw
        else:
            # Generate synthetic IMU data
            t = time.time()
            yaw = 30 * math.sin(0.5 * t)
            pitch = 10 * math.sin(0.3 * t)
            roll = 0.0
            # Convert to quaternion (simplified)
            qw = math.cos(math.radians(yaw)/2) * math.cos(math.radians(pitch)/2)
            qx = math.sin(math.radians(yaw)/2) * math.cos(math.radians(pitch)/2)
            qy = 0.0
            qz = math.cos(math.radians(yaw)/2) * math.sin(math.radians(pitch)/2)
            return [qw, qx, qy, qz, roll, pitch, yaw]

    def get_next_vision(self):
        if self.vision_data:
            if self.vision_index >= len(self.vision_data):
                self.vision_index = 0  # loop
            row = self.vision_data[self.vision_index]
            self.vision_index += 1
            return [float(row[1]), float(row[2]), float(row[3]), float(row[4]), row[5]]  # az,el,dist,conf,cls
        else:
            # Generate synthetic vision data
            t = time.time()
            az = 45 * math.sin(0.2 * t)
            el = 15 * math.cos(0.2 * t)
            dist = 2.0 + 1.0 * math.sin(0.1 * t)
            conf = 0.8
            cls_name = "person"
            return [az, el, dist, conf, cls_name]