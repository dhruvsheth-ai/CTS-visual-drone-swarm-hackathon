#!/usr/bin/python3

import numpy as np
import cv2
import onnxruntime as ort

import sys
import time
import traceback

import av
import djitellopy
import math

# import rospy
# from geometry_msgs.msg import PoseStamped
import threading
# from helper_functions import *

## Global Variable ##
position_drone_1 = np.zeros((1, 3), dtype=np.float32)
orientation_drone_1 = np.zeros((1, 4), dtype=np.float32)

goal_position = np.zeros((1, 3), dtype=np.float32)

obs_1_data = np.zeros((1, 14), dtype=np.float32)
recurrent_data = np.zeros((1, 1, 256), dtype=np.float32)

telloID1 = "tello3"

frame_count = 0

## ONNX Model for Drone Decision Making ##
model = "../model/VisualDrone_single_drone.onnx"
sess = ort.InferenceSession(model)

obs_0 = sess.get_inputs()[0].name
obs_1 = sess.get_inputs()[1].name
recurrent_in = sess.get_inputs()[2].name

## Handlers and control functions ##


def scale_vel_cmd(cmd_val):
    SCALE = 0.3

    return SCALE * cmd_val


def cb_cmd_vel(drone, continuous_action):

    linear_z, linear_x, linear_y, angular_z = continuous_action[0]

    drone.set_pitch(scale_vel_cmd(linear_y))
    drone.set_roll(scale_vel_cmd(linear_x))
    drone.set_yaw(scale_vel_cmd(angular_z))
    drone.set_throttle(scale_vel_cmd(linear_z))


def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        pass


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


def position_update_thread(drone):
    global position_drone_1, orientation_drone_1, goal_position

    try:
        while True:

            msg_orientation = drone.query_attitude()
            [qw, qx, qy, qz] = euler_to_quaternion(msg_orientation["roll"],
                                                   msg_orientation["pitch"],
                                                   msg_orientation["yaw"])
            orientation_drone_1[0, 0] = qx
            orientation_drone_1[0, 1] = qy
            orientation_drone_1[0, 2] = qz
            orientation_drone_1[0, 3] = qw

            print(orientation_drone_1)

            # INSERT POSITION UPDATE HERE

            time.sleep(2)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        print("problem")
        # drone.end()
        # cv2.destroyAllWindows()


def main():
    global obs_1_data, recurrent_data
    global position_drone_1, orientation_drone_1, goal_position
    global frame_count

    drone = djitellopy.Tello()
    t = threading.Thread(target=position_update_thread, args=(drone, ))
    t.start()
    try:
        # drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        drone.connect(wait_for_state=True)
        # drone.wait_for_connection(60.0)

        drone.takeoff()

        time.sleep(5)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
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

                image = np.array(frame.to_image())[:, 120:839, :]

                processed_frame = np.array(image_processing(image),
                                           dtype=np.float32)

                cv2.imshow('Original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                _, _, continuous_action, _, recurrent_data = sess.run(
                    None, {
                        obs_0: processed_frame,
                        obs_1: obs_1_data,
                        recurrent_in: recurrent_data
                    })

                mul = np.matmul(
                    quarternion_to_rotation_matrix(orientation_drone_1[0]),
                    UNIT_VECTOR_X)
                normalized_mul = mul / np.linalg.norm(mul)

                obs_1_data[0, 0:3] = position_drone_1
                obs_1_data[0, 3:7] = orientation_drone_1
                obs_1_data[0, 7:10] = np.reshape(normalized_mul, (1, 3))
                obs_1_data[0, 10:14] = continuous_action

                cb_cmd_vel(drone, continuous_action)

                distance_target = np.linalg.norm(position_drone_1 -
                                                 goal_position)

                if distance_target < 0.5:
                    cb_cmd_vel(drone, [[0, 0, 0, 0]])
                    drone.land()
                    print("End Frame:", frame_count)
                    print("End Time:", time.time() - start_time_all)
                    time.sleep(5)
                    drone.end()
                    cv2.destroyAllWindows()

                ####################

                if frame.time_base < 1.0 / 60:
                    time_base = 1.0 / 60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.end()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # rospy.init_node('test', anonymous=True)
    # rospy.Subscriber(f'/vrpn_client_node/{telloID1}/pose', PoseStamped, current_positioning_drone_1)
    # rospy.Subscriber(f'/vrpn_client_node/goal/pose', PoseStamped, goal_positioning)

    main()
