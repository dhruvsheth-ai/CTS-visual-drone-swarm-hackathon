#!/usr/bin/python3
import numpy as np
import cv2
import onnxruntime as ort
import math
import sys
import time
import traceback
import threading
import av
# import tellopy

# import rospy
# from geometry_msgs.msg import PoseStamped

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


def position_update_thread(drone):
    global position_drone_1, orientation_drone_1, goal_position
    #time.sleep(2)
    try:
        while True:

            msg_orientation = drone.query_attitude()
            print(msg_orientation)
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

    # Scale the inputs from -1 to 1 range to -100 to 100 range for djitellopy
    # Assuming the original scaling factor of 0.3 is to limit the maximum speed,
    # you might want to adjust the scaling to fit the -100 to 100 range.
    # For example, if you still want to limit the speed, you could use a scaling factor
    # that considers the max value of 100. Here, I'm directly scaling to the full range.

    scaled_linear_y = int(linear_y * 100)  # Forward/Backward
    scaled_linear_x = int(linear_x * 100)  # Left/Right
    scaled_angular_z = int(angular_z * 100)  # Yaw
    scaled_linear_z = int(linear_z * 100)  # Up/Down

    # Use the send_rc_control method to send the scaled commands to the drone
    drone.send_rc_control(scaled_linear_x, scaled_linear_y, scaled_linear_z,
                          scaled_angular_z)


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

        # tello.takeoff()
        #drone.takeoff()

        time.sleep(5)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                tello.streamon()
                frame = tello.get_frame_read()
                container = frame
            except av.AVError as ave:
                print(ave)
                print('retry...')
                pass

        # skip first 300 frames
        frame_skip = 300
        time.sleep(2)  # Wait for the video stream to stabilize

        # t = threading.Thread(target=position_update_thread, args=(tello, ))
        # t.daemon = True
        # t.start()

        while True:
            position_update_thread(tello)

            frame = container.frame
            if frame is not None:
                start_time = time.time()
                #
                ####################

                if frame_count == 0:
                    start_time_all = time.time()

                frame_count += 1

                image = np.array(frame)[:, 120:839, :]

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

                cb_cmd_vel(tello, continuous_action)

                distance_target = np.linalg.norm(position_drone_1 -
                                                 goal_position)

                if distance_target < 0.5:
                    cb_cmd_vel(tello, [[0, 0, 0, 0]])
                    # tello.land()
                    print("End Frame:", frame_count)
                    print("End Time:", time.time() - start_time_all)
                    time.sleep(5)
                    #drone.quit()
                    cv2.destroyAllWindows()

                ####################

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':

    main()
