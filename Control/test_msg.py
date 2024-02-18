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
            print(drone.query_attitude())
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
