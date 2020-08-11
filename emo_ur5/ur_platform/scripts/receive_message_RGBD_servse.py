import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import FollowJointTrajectoryActionFeedback
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import PoseArray, Quaternion, PoseStamped
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
import message_filters
import cv2
import time
import numpy as np
import h5py

record_robot_state_arm_gripper_velocity = []  # Dimensions of the matrix is [number of samples step]
storage_robot_state_arm_gripper_velocity = []  # Dimensions of the matrix is [number of samples]

record_robot_state_arm_gripper_position = []  # Dimensions of the matrix is [number of samples step]
storage_robot_state_arm_gripper_position = []  # Dimensions of the matrix is [number of samples]

record_robot_state_image = []  # Dimensions of the image is [800, 800, 3]
storage_robot_state_image = []

record_object_state = []  # Dimensions of the image is [step, 3]
storage_object_state = []

success_num = 0
record_num = 2
have_record = 0


def write_file(txt='string'):
    f = open('tmp.txt', 'w')
    f.write(txt)
    f.close()


def write_file_record(txt='string'):
    f = open('record_state.txt', 'w')
    f.write(txt)
    f.close()


def read_file():
    f = open('tmp.txt')
    return f.readline()


def get_object_position(record):
    object = rospy.wait_for_message('gazebo/model_states', ModelStates)
    for i in range(len(object.name)):
        if object.name[i] == 'object':
            object_set = ModelState()
            object_set.model_name = object.name[i]
            object_set.pose = object.pose[i]
            x = object_set.pose.position.x
            y = object_set.pose.position.y
            z = object_set.pose.position.z
    if record == False:
        return x, y
    else:
        return [x, y, z]


def storage_data(done):
    global record_robot_state_arm_gripper_velocity, storage_robot_state_arm_gripper_velocity, record_robot_state_arm_gripper_position, storage_robot_state_arm_gripper_position, record_robot_state_image, storage_robot_state_image
    global record_object_state, storage_object_state
    if done == 'Success':
        storage_robot_state_arm_gripper_velocity.append(record_robot_state_arm_gripper_velocity)
        storage_robot_state_arm_gripper_position.append(record_robot_state_arm_gripper_position)
        storage_robot_state_image.append(record_robot_state_image)
        storage_object_state.append(record_object_state)
        record_robot_state_arm_gripper_velocity = []
        record_robot_state_arm_gripper_position = []
        record_robot_state_image = []
        record_object_state = []
    else:
        record_robot_state_arm_gripper_velocity = []
        record_robot_state_arm_gripper_position = []
        record_robot_state_image = []
        record_object_state = []


def storage_data_to_h5(storage_robot_state_arm_gripper_velocity, storage_robot_state_arm_gripper_position,
                       storage_robot_state_image, storage_object_state):
    write_file_record('writing')
    storage_robot_state_arm_gripper_velocity = np.array(storage_robot_state_arm_gripper_velocity)
    storage_robot_state_arm_gripper_position = np.array(storage_robot_state_arm_gripper_position)
    storage_robot_state_image = np.array(storage_robot_state_image)
    storage_object_state = np.array(storage_object_state)
    with h5py.File("storage_date_100_130_fast_2.hdf5", 'w') as f:
        for i in range(len(storage_robot_state_image)):
            storage_robot_state_arm_gripper_name_velocity = "storage_robot_state_arm_gripper_velocity" + str(i)
            storage_robot_state_arm_gripper_name_position = "storage_robot_state_arm_gripper_position" + str(i)
            storage_robot_state_image_name = "storage_robot_state_image" + str(i)
            storage_object_state_name = "storage_object_state" + str(i)
            f.create_dataset(storage_robot_state_arm_gripper_name_velocity,
                             data=storage_robot_state_arm_gripper_velocity[i],
                             compression="gzip",
                             compression_opts=7)
            f.create_dataset(storage_robot_state_arm_gripper_name_position,
                             data=storage_robot_state_arm_gripper_position[i],
                             compression="gzip",
                             compression_opts=7)
            f.create_dataset(storage_robot_state_image_name, data=storage_robot_state_image[i], compression="gzip",
                             compression_opts=7)
            f.create_dataset(storage_object_state_name, data=storage_object_state[i], compression="gzip",
                             compression_opts=7)
    write_file_record('over_writing')


def call_back(image, depth):
    global record_robot_state_arm_gripper_velocity, storage_robot_state_arm_gripper_velocity, record_robot_state_arm_gripper_position, storage_robot_state_arm_gripper_position, record_robot_state_image, storage_robot_state_image, success_num
    global record_object_state, storage_object_state
    global have_record
    robot_state = read_file()
    if robot_state == 'Start':
        joint = rospy.wait_for_message('/joint_states', JointState, timeout=5)
        if sum(map(abs, joint.velocity)) < 0.060:
            print "arm is stopping, break"
            return
        object_state = get_object_position(True)
        robot_state_cv_depth = CvBridge().imgmsg_to_cv2(depth, "passthrough")
        robot_state_cv_depth = cv2.normalize(robot_state_cv_depth, robot_state_cv_depth, 0, 255,
                                             cv2.NORM_MINMAX)
        robot_state_cv_depth = np.expand_dims(robot_state_cv_depth, 2)
        robot_state_cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
        robot_state_cv_image = np.append(robot_state_cv_image, robot_state_cv_depth, axis=2)
        robot_state_arm_gripper = joint
        record_object_state.append(object_state)
        record_robot_state_arm_gripper_velocity.append(list(robot_state_arm_gripper.velocity))
        record_robot_state_arm_gripper_position.append(list(robot_state_arm_gripper.position))
        record_robot_state_image.append(robot_state_cv_image)
        print 'ready'
        print len(record_robot_state_image)
    elif robot_state == 'Finish':
        get_object_x, get_object_y = get_object_position(False)
        if 0.4 < get_object_y - 0.2 < 0.6 and 0.3 < get_object_x < 0.54:
            storage_data('Success')
            success_num += 1
        else:
            storage_data('Failure')
        write_file('Wait')
    if success_num % record_num == 0 and have_record != success_num and success_num != 0:
        storage_data_to_h5(storage_robot_state_arm_gripper_velocity, storage_robot_state_arm_gripper_position,
                           storage_robot_state_image, storage_object_state)
        have_record = success_num
        print 'success record data'


if __name__ == '__main__':
    write_file('1')
    rospy.init_node('receive_message', anonymous=True)
    subsciber_images = message_filters.Subscriber('/kinect/rgb/image_raw', Image)
    subsciber_depth = message_filters.Subscriber('/kinect/depth/image_raw', Image)
    # subsciber_joint = message_filters.Subscriber('/joint_states', JointState)
    ts = message_filters.TimeSynchronizer([subsciber_images, subsciber_depth], 1)
    ts.registerCallback(call_back)
    rospy.spin()
