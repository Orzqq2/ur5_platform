import numpy as np
import random
import tensorflow as tf
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from control_msgs.msg import FollowJointTrajectoryActionFeedback
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelConfiguration
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import String, Float64, Bool, Float64MultiArray
from geometry_msgs.msg import PoseArray, Quaternion, PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty
import cv2
import time
import numpy as np
import time

from data_generator import DataGenerator
from mil_deep_images_coordinate import MIL
from tensorflow.python.platform import flags

update_batch_size = 5
test_batch_size = 3
meta_batch_size = 3
num_update = 3
use_meta_learning = False
flag_train = True
log_dir = './Save/model_1'


def get_object_position():
    object = rospy.wait_for_message('gazebo/model_states', ModelStates)
    for i in range(len(object.name)):
        if object.name[i] == 'object':
            object_set = ModelState()
            object_set.model_name = object.name[i]
            object_set.pose = object.pose[i]
            x = object_set.pose.position.x
            y = object_set.pose.position.y
    return x, y


def reset_arm():
    reset_ur5 = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
    reset_ur5(model_name='robot', urdf_param_name='',
              joint_names=['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                           'wrist_2_joint', 'wrist_3_joint'], joint_positions=[0, -0.427, 0.976, -0.454, 0, -0.0511])
    reset_ur5(model_name='robot', urdf_param_name='',
              joint_names=['gripper_finger1_joint'], joint_positions=[0.02])


def switch_controller(state='Stop'):
    sw_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
    if state in 'Stop_':
        sw_controller(start_controllers=['arm_controller_stop'], stop_controllers=['arm_controller'], strictness=1)
    else:
        sw_controller(start_controllers=['arm_controller'], stop_controllers=['arm_controller_stop'], strictness=1)


def reset_world(reset_proxy):
    rospy.wait_for_service('/gazebo/reset_world')
    try:
        # reset_proxy.call()
        reset_proxy()
    except (rospy.ServiceException) as e:
        print ("/gazebo/reset_simulation service call failed")


def train(graph, model, saver, sess, data_generator, meta=True):
    """
    Train the model.
    """
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    TOTAL_ITERS = 50000
    prelosses, postlosses = [], []

    if meta == True:
        for itr in range(TOTAL_ITERS):
            train_batch_image, train_batch_joint, train_batch_action = data_generator.generate_data_batch(itr)
            feed_dict = {model.statea_image: train_batch_image[:, :update_batch_size, :, :, :],
                         model.statea_joint: train_batch_joint[:, :update_batch_size, :],
                         model.actiona: train_batch_action[:, :update_batch_size, :],
                         model.stateb_image: train_batch_image[:, update_batch_size:, :, :, :],
                         model.stateb_joint: train_batch_joint[:, update_batch_size:, :],
                         model.actionb: train_batch_action[:, update_batch_size:, :]}
            input_tensors = [model.train_op, model.total_losses2, model.test_output_action]
            # if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            #     input_tensors.extend([model.train_summ_op, model.total_losses2[model.num_updates - 1]])
            with graph.as_default():
                results = sess.run(input_tensors, feed_dict=feed_dict)
            print results[1]
        print 'Saving model'
        with graph.as_default():
            saver.save(sess, log_dir)
    else:
        for itr in range(TOTAL_ITERS):
            train_batch_image, train_batch_joint, train_batch_action, train_batch_object = data_generator.generate_data_batch_nn(
                itr)
            feed_dict = {model.statea_image: train_batch_image,
                         model.statea_joint: train_batch_joint,
                         model.actiona: train_batch_action,
                         model.object: train_batch_object}
            input_tensors = [model.train_op, model.lossa, model.arm_action]
            with graph.as_default():
                results = sess.run(input_tensors, feed_dict=feed_dict)
            print results[1]
            print results[2][20]
            print train_batch_action[20]
        print 'Saving model'
        with graph.as_default():
            saver.save(sess, log_dir)


def test_gazebo(graph, model, saver, sess, meta=True):
    rospy.init_node('control_ur5', anonymous=True)
    reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    vel_arm_pub = rospy.Publisher('/arm_controller/command', Float64MultiArray, queue_size=5)
    vel_gripper_pub = rospy.Publisher('/gripper_controller/command', Float64, queue_size=5)
    arm_action, grip_action = Float64MultiArray(), Float64()
    # reset_world(reset_proxy)
    reset_arm()
    arm_action.data = [0, -0.045, 0, 0, 0, 0]
    switch_controller('Stop')
    grip_action = 0
    step = 0
    success_num = 0
    while True:
        if meta == True:
            pass
        else:
            switch_controller('Start')
            # Get UR5 state and images
            # rospy.wait_for_service('/gazebo/unpause_physics')
            # try:
            #     unpause()
            # except (rospy.ServiceException) as e:
            #     print ("/gazebo/unpause_physics service call failed")
            vel_arm_pub.publish(arm_action)
            vel_gripper_pub.publish(grip_action)
            time.sleep(0.05)
            robot_state_arm_gripper = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            robot_state_depth = rospy.wait_for_message('/kinect/depth/image_raw', Image, timeout=5)
            tmp = 0
            depth_tmp = robot_state_depth.header.stamp.nsecs / 1000000000.0 + float(
                robot_state_depth.header.stamp.secs)
            while tmp < depth_tmp:
                robot_state_image = rospy.wait_for_message('/camera/image_raw', Image, timeout=5)
                tmp = robot_state_image.header.stamp.nsecs / 1000000000.0 + float(
                    robot_state_image.header.stamp.secs)
            if tmp - depth_tmp > 0.12:
                robot_state_depth = rospy.wait_for_message('/kinect/depth/image_raw', Image, timeout=5)
            robot_state_cv_depth = CvBridge().imgmsg_to_cv2(robot_state_depth, "passthrough")
            # rospy.wait_for_service('/gazebo/pause_physics')
            # try:
            #     # resp_pause = pause.call()
            #     pause()
            # except (rospy.ServiceException) as e:
            #     print ("/gazebo/pause_physics service call failed")
            # Finish round
            switch_controller('Stop')
            get_object_x, get_object_y = get_object_position()
            if 0.4 < get_object_y - 0.2 < 0.6 and 0.3 < get_object_x < 0.54:
                success_num += 1
                # reset_world(reset_proxy)
                reset_arm()
            elif step >= 10000:
                # reset_world(reset_proxy)
                reset_arm()
            step += 1

            robot_state_cv_depth = cv2.normalize(robot_state_cv_depth, robot_state_cv_depth, 0, 255,
                                                 cv2.NORM_MINMAX)
            robot_state_cv_depth = np.expand_dims(robot_state_cv_depth, 2)
            robot_state_cv_image = CvBridge().imgmsg_to_cv2(robot_state_image, "bgr8")
            robot_state_cv_image = np.append(robot_state_cv_image, robot_state_cv_depth, axis=2)

            feed_dict = {model.statea_image: np.expand_dims(robot_state_cv_image, axis=0),
                         model.statea_joint: np.expand_dims(robot_state_arm_gripper.position, axis=0)}

            with graph.as_default():
                arm_action_data, grip_action = sess.run([model.arm_action, model.grip_action], feed_dict=feed_dict)
            print arm_action_data[0]
            arm_action.data = list(arm_action_data[0])
            arm_action.data[1] = arm_action.data[1] - 0.045


def main():
    tf.set_random_seed(1)
    np.random.seed(1)
    random.seed(1)
    # Test environment
    # if not FLAGS.train:
    #     if 'reach' in FLAGS.experiment:
    #         env = gym.make('ReacherMILTest-v1')
    #         ob = env.reset()
    # import pdb; pdb.set_trace()
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)

    # Create object
    if flag_train:
        data_generator = DataGenerator(update_batch_size, test_batch_size, meta_batch_size)
    model = MIL(num_update)

    # put here for now
    if flag_train:
        if use_meta_learning:
            model.init_network(graph)
        else:
            model.init_network(graph, meta_learning='False')
    else:
        model.init_network(graph, prefix='Testing', meta_learning='False')
    with graph.as_default():
        # Set up saver.
        saver = tf.train.Saver(max_to_keep=10)
        # Initialize variables.
        init_op = tf.global_variables_initializer()
        sess.run(init_op, feed_dict=None)
        # Start queue runners (used for loading videos on the fly)
        tf.train.start_queue_runners(sess=sess)
    if flag_train == False:
        # Restore model from file
        with graph.as_default():
            try:
                saver.restore(sess, log_dir)
                print 'load'
            except:
                init_op = tf.global_variables_initializer()
                sess.run(init_op, feed_dict=None)
    if flag_train:
        train(graph, model, saver, sess, data_generator, meta=use_meta_learning)
    else:
        test_gazebo(graph, model, saver, sess, meta=use_meta_learning)


if __name__ == "__main__":
    main()
