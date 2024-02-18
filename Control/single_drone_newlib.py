#!/usr/bin/python3
import numpy as np
import cv2
import onnxruntime as ort

import sys
import time
import traceback

import av
import tellopy

import rospy
from geometry_msgs.msg import PoseStamped

from djitellopy import Tello

from helper_functions import *

## Global Variable ##
position_drone_1 = np.zeros((1, 3), dtype=np.float32)
orientation_drone_1 = np.zeros((1, 4), dtype=np.float32)

goal_position = np.zeros((1, 3), dtype=np.float32)

obs_1_data = np.zeros((1, 14), dtype=np.float32)
recurrent_data = np.zeros((1, 1, 256), dtype=np.float32)

telloID1 = "tello3"

frame_count = 0

## ONNX Model for Drone Decision Making ##
model = "../Models/VisualDrone_single_drone.onnx"
sess = ort.InferenceSession(model)

obs_0 = sess.get_inputs()[0].name
obs_1 = sess.get_inputs()[1].name
recurrent_in = sess.get_inputs()[2].name

## Handlers and control functions ##


def set_throttle(self, throttle):
    """
    Set_throttle controls the vertical up and down motion of the drone.
    Pass in an int from -1.0 ~ 1.0. (positive value means upward)
    """
    self.left_y = self.__fix_range(throttle)


def set_yaw(self, yaw):
    """
    Set_yaw controls the left and right rotation of the drone.
    Pass in an int from -1.0 ~ 1.0. (positive value will make the drone turn to the right)
    """
    self.left_x = self.__fix_range(yaw)

def set_pitch(self, pitch):
    """
    Set_pitch controls the forward and backward tilt of the drone.
    Pass in an int from -1.0 ~ 1.0. (positive value will make the drone move forward)
    """
    self.right_y = self.__fix_range(pitch)


def set_roll(self, roll):
    """
    Set_roll controls the the side to side tilt of the drone.
    Pass in an int from -1.0 ~ 1.0. (positive value will make the drone move to the right)
    """
    self.right_x = self.__fix_range(roll)










def current_positioning_drone_1(msg):
    global position_drone_1, orientation_drone_1

    position_drone_1[0, 0] = msg.pose.position.x
    position_drone_1[0, 1] = msg.pose.position.z
    position_drone_1[0, 2] = msg.pose.position.y

    orientation_drone_1[0, 0] = msg.pose.orientation.x
    orientation_drone_1[0, 1] = msg.pose.orientation.y
    orientation_drone_1[0, 2] = msg.pose.orientation.z
    orientation_drone_1[0, 3] = msg.pose.orientation.w


def goal_positioning(msg):
    global goal_position

    goal_position[0, 0] = msg.pose.position.x
    goal_position[0, 1] = msg.pose.position.z
    goal_position[0, 2] = msg.pose.position.y



def cb_cmd_vel(drone, continuous_action):

    linear_z, linear_x, linear_y, angular_z = continuous_action[0]



    drone.move_left(0.3 * (linear_y) * 100)
    drone.move_forward(0.3 * (linear_x) * 100)
    drone.move_down(0.3 * (angular_z) * 100)
    drone.set_speed(0.3 * (linear_z) * 100)

def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        pass
        
def main():
    global obs_1_data, recurrent_data
    global position_drone_1, orientation_drone_1, goal_position
    global frame_count

    #drone = tellopy.Tello()
    tello = Tello()

    try:
        tello.connect()
        #drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        #drone.connect()
        #drone.wait_for_connection(60.0)

        tello.takeoff()
        #drone.takeoff()

        time.sleep(5)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                tello.streamon()
                container = av.open(tello.get_frame_read())
            except av.AVError as ave:
                print(ave)
                print('retry...')
                pass

        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()

                ####################

                if frame_count == 0:
                    start_time_all = time.time()

                frame_count += 1

                image = np.array(frame.to_image())[:, 120: 839, :]

                processed_frame = np.array(image_processing(image), dtype=np.float32)

                cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                _, _, continuous_action, _,  recurrent_data = sess.run(None, {
                    obs_0: processed_frame,
                    obs_1: obs_1_data,
                    recurrent_in: recurrent_data
                })

                mul = np.matmul(
                    quarternion_to_rotation_matrix(orientation_drone_1[0]),
                    UNIT_VECTOR_X
                )
                normalized_mul = mul / np.linalg.norm(mul)

                obs_1_data[0, 0: 3] = position_drone_1
                obs_1_data[0, 3: 7] = orientation_drone_1
                obs_1_data[0, 7: 10] = np.reshape(normalized_mul, (1, 3))
                obs_1_data[0, 10: 14] = continuous_action

                cb_cmd_vel(drone, continuous_action)

                distance_target = np.linalg.norm(position_drone_1 - goal_position)

                if distance_target < 0.5:
                    cb_cmd_vel(tello, [[0, 0, 0, 0]])
                    tello.land()
                    print("End Frame:", frame_count)
                    print("End Time:", time.time() - start_time_all)
                    time.sleep(5)
                    #drone.quit()
                    cv2.destroyAllWindows()

                ####################

                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    rospy.Subscriber(f'/vrpn_client_node/{telloID1}/pose', PoseStamped, current_positioning_drone_1)
    rospy.Subscriber(f'/vrpn_client_node/goal/pose', PoseStamped, goal_positioning)

    main()
