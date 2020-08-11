import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import FollowJointTrajectoryActionFeedback
from gazebo_msgs.msg import ModelState, ModelStates
from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import PoseArray, Quaternion, PoseStamped
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import h5py


class ReceiveROSMessage(object):
    def __init__(self):
        super(ReceiveROSMessage, self).__init__()

        ## Initialize rospy node and CV
        rospy.init_node('receive_message', anonymous=True)
        self.bridge = CvBridge()

        # [elbow_joint, gripper_finger1_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint,
        # wrist_2_joint, wrist_3_joint]
        self.record_robot_state_arm_gripper_velocity = []  # Dimensions of the matrix is [number of samples step]
        self.storage_robot_state_arm_gripper_velocity = []  # Dimensions of the matrix is [number of samples]

        self.record_robot_state_arm_gripper_position = []  # Dimensions of the matrix is [number of samples step]
        self.storage_robot_state_arm_gripper_position = []  # Dimensions of the matrix is [number of samples]

        self.record_robot_state_image = []  # Dimensions of the image is [800, 800, 3]
        self.storage_robot_state_image = []

        self.success_num = 0

    def get_object_position(self):
        object = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(object.name)):
            if object.name[i] == 'object':
                object_set = ModelState()
                object_set.model_name = object.name[i]
                object_set.pose = object.pose[i]
                x = object_set.pose.position.x
                y = object_set.pose.position.y
        return x, y

    def receive_state(self):
        robot_state_arm_gripper = None
        robot_state_image = None
        print 'ready'
        while robot_state_arm_gripper is None:
            try:
                # robot_state_arm = rospy.wait_for_message('/arm_controller/follow_joint_trajectory/feedback',
                #                                          FollowJointTrajectoryActionFeedback, timeout=5)

                robot_state_arm_gripper = rospy.wait_for_message('/joint_states', JointState, timeout=5)
                if sum(map(abs, robot_state_arm_gripper.velocity)) < 0.05:
                    print "arm is stopping, break"
                    break
                robot_state_image = rospy.wait_for_message('/kinect/rgb/image_raw', Image, timeout=5)
                robot_state_cv_image = self.bridge.imgmsg_to_cv2(robot_state_image, "bgr8")
                self.record_robot_state_arm_gripper_velocity.append(list(robot_state_arm_gripper.velocity))
                self.record_robot_state_arm_gripper_position.append(list(robot_state_arm_gripper.position))
                self.record_robot_state_image.append(robot_state_cv_image)
            except:
                pass

    def storage_data(self, done):
        if done == 'Success':
            self.storage_robot_state_arm_gripper_velocity.append(self.record_robot_state_arm_gripper_velocity)
            self.storage_robot_state_arm_gripper_position.append(self.record_robot_state_arm_gripper_position)
            self.storage_robot_state_image.append(self.record_robot_state_image)
            self.record_robot_state_arm_gripper_velocity = []
            self.record_robot_state_arm_gripper_position = []
            self.record_robot_state_image = []
        else:
            self.record_robot_state_arm_gripper_velocity = []
            self.record_robot_state_arm_gripper_position = []
            self.record_robot_state_image = []

    def write_file(self, txt='string'):
        f = open('tmp.txt', 'w')
        f.write(txt)
        f.close()

    def write_file_record(self, txt='string'):
        f = open('record_state.txt', 'w')
        f.write(txt)
        f.close()

    def read_file(self):
        f = open('tmp.txt')
        return f.readline()

    def judgment_status(self):
        robot_state = self.read_file()
        if robot_state == 'Start':
            self.receive_state()
        elif robot_state == 'Finish':
            get_object_x, get_object_y = self.get_object_position()
            if 0.4 < get_object_y - 0.2 < 0.6 and 0.3 < get_object_x < 0.54:
                self.storage_data('Success')
                self.success_num += 1
            else:
                self.storage_data('Failure')
            self.write_file('Wait')

    def storage_data_to_h5(self):
        self.write_file_record('writing')
        storage_robot_state_arm_gripper_velocity = np.array(self.storage_robot_state_arm_gripper_velocity)
        storage_robot_state_arm_gripper_position = np.array(self.storage_robot_state_arm_gripper_position)
        storage_robot_state_image = np.array(self.storage_robot_state_image)
        with h5py.File("storage_date_100_3.hdf5", 'w') as f:
            for i in range(len(self.storage_robot_state_image)):
                storage_robot_state_arm_gripper_name_velocity = "storage_robot_state_arm_gripper_velocity" + str(i)
                storage_robot_state_arm_gripper_name_position = "storage_robot_state_arm_gripper_position" + str(i)
                storage_robot_state_image_name = "storage_robot_state_image" + str(i)
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
        self.write_file_record('over_writing')


def main():
    test = ReceiveROSMessage()
    test.write_file('1')
    record_num = 10
    have_record = 0
    while True:
        test.judgment_status()
        if test.success_num % record_num == 0 and have_record != test.success_num and test.success_num != 0:
            test.storage_data_to_h5()
            have_record = test.success_num
            print 'success record data'


if __name__ == '__main__':
    main()
