#!/usr/bin/python3

import numpy as np
import cv2
import onnxruntime as ort

import sys
import time
import traceback

#import av
#import tellopy
import djitellopy
# import rospy
# from geometry_msgs.msg import PoseStamped

from helper_functions import *

import math

position_drone_1 = np.zeros((1, 3), dtype=np.float32)
orientation_drone_1 = np.zeros((1, 4), dtype=np.float32)

goal_position = np.zeros((1, 3), dtype=np.float32)


def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(
        yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(
        yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(
        yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(
        yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)

    return [qw, qx, qy, qz]


def main():
    global obs_1_data, recurrent_data
    global position_drone_1, orientation_drone_1, goal_position
    global frame_count

    drone = djitellopy.Tello()

    try:

        drone.connect(wait_for_state=True)
        # drone.wait_for_connection(60.0)

        #         drone.takeoff()

        #       time.sleep(5)

        # skip first 300 frames
        while True:
            global position_drone_1, orientation_drone_1

            msg_orientation = drone.query_attitude()
            print(msg_orientation)
            position_drone_1[0, 0] = 0
            position_drone_1[0, 1] = 0
            position_drone_1[0, 2] = 0

            [qw, qx, qy, qz] = euler_to_quaternion(msg_orientation["roll"],
                                                   msg_orientation["pitch"],
                                                   msg_orientation["yaw"])
            orientation_drone_1[0, 0] = qx
            orientation_drone_1[0, 1] = qy
            orientation_drone_1[0, 2] = qz
            orientation_drone_1[0, 3] = qw

            print(orientation_drone_1)

            time.sleep(2)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
