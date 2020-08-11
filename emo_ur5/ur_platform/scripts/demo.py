#!/usr/bin/env python

import sys
import copy
import rospy
import rospkg
import os
import tty
import termios
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import tf
import time
import onrobot_rg2
import yaml
from moveit_msgs.srv import GetPlanningScene, GetPositionIK
from geometry_msgs.msg import PoseArray, Quaternion, PoseStamped
from ur_msgs.srv import SetIO
from tf.transformations import *
from math import pi
from std_msgs.msg import String, Float64, Bool
from moveit_commander.conversions import pose_to_list
from epicker.srv import *
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_script import URScript

# Set up path to load environment models
script_dir = os.path.dirname(__file__)
platform_model_rel_path = "../model/"
platform_model_path = os.path.join(script_dir, platform_model_rel_path)

# Keyboard input
i = sys.stdin.fileno()
o = sys.stdout.fileno()
backup = termios.tcgetattr(i)
#key = 0

flag_sim = False # Simulation flag
flag_pp = False # Use the pick() and place() function
io_control = False
flag_ompl = False
read_traj = True

pose_offset_x = 0
pose_offset_y = 0
pose_offset_z = 0

rospack = rospkg.RosPack()
g_path2package = rospack.get_path('epicker')
traj_path = g_path2package + '/saved_trajectories/'

#print "----------> argv len: %s"%len(sys.argv)
if len(sys.argv) is 3: # Not simulation #TODO: modify this method
    flag_sim = False

def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
      for index in range(len(goal)):
        if abs(actual[index] - goal[index]) > tolerance:
          return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
      return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
      return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True

class PickNPlace(object):
    def __init__(self):
        super(PickNPlace, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('pick_and_place_node', anonymous=True)

        ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
        ## the robot:
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
        ## to the world surrounding the robot:
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)

        ## Instantiate a `MoveGroupCommander`_ object.
        group_name = "manipulator"
        group = moveit_commander.MoveGroupCommander(group_name)

        ## We create a `DisplayTrajectory`_ publisher which is used later to publish
        ## trajectories for RViz to visualize:
        '''
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)
        '''

        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = group.get_planning_frame()
        print "============ Reference frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = group.get_end_effector_link()
        print "============ End effector: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print "============ Robot Groups:", robot.get_group_names()

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print "============ Printing robot state"
        print robot.get_current_state()
        print ""

        # Set the work space
        ws = [-0.87,-0.33,-0.4,0.31,0.8,0.9]
        group.set_workspace(ws)

        # Set planner
        group.allow_replanning(True)

        if flag_ompl:
            group.set_planner_id('RRTstarkConfigDefault')
            group.set_planning_time(1.5)
            group.allow_looking(True)
            group.set_num_planning_attempts(6)
        else:
            group.set_planning_time(1.5)
            group.set_num_planning_attempts(6)

        # Allow some leeway in position (meters) and orientation (radians)
        group.set_goal_position_tolerance(0.005)
        group.set_goal_orientation_tolerance(0.05)

        # Set moving speed scaling factor
        group.set_max_velocity_scaling_factor(0.5)

        # Misc variables
        self.robot = robot
        self.scene = scene
        self.group = group
        #self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.graspPoses = []
        self.preGraspPoses = []
        self.pre_grasp_dis = 0.30
        self.grasp_dis = 0.20
        self.move_up_dis = 0.03
        self.order = {}
        self.pickup_list = []
        self.grasps = []
        self.obj_sizes = {'JDB': params["dimensions"]['JDB'], 'YZ': params["dimensions"]['YZ']}

    def load_environment(self):
        self.add_table()
        #self.add_front_wall()
        #self.add_back_wall()
        #self.add_ceilling()
        #self.add_camera()
        self.add_base()
        self.load_bookshelf()
        self.load_bin()
        return

    def load_bookshelf(self):
        mesh_name = "bookshelf"
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 0.5
        platform_pose.pose.orientation.x = 0.5
        platform_pose.pose.orientation.y = 0.5
        platform_pose.pose.orientation.z = 0.5
        platform_pose.pose.position.x = -0.28
        platform_pose.pose.position.y = 0.21
        platform_pose.pose.position.z = -0.415

        self.scene.add_mesh(mesh_name, platform_pose, platform_model_path+"cabinet.stl", size=(1,1,1))
        return self.wait_for_state_update(box_name=mesh_name, box_is_known=True, box_is_attached=False)

    def load_bin(self):
        mesh_name = "bin"
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 0.5
        platform_pose.pose.orientation.x = 0.5
        platform_pose.pose.orientation.y = 0.5
        platform_pose.pose.orientation.z = 0.5
        platform_pose.pose.position.x = -0.28
        platform_pose.pose.position.y = 0.22
        platform_pose.pose.position.z = -0.39

        self.scene.add_mesh(mesh_name, platform_pose, platform_model_path+"bin.stl", size=(1,1,1))
        return self.wait_for_state_update(box_name=mesh_name, box_is_known=True, box_is_attached=False)

    def add_table(self, timeout=4):
        box_name = "table"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = -0.46
        scene.add_box(box_name, box_pose, size=(2.5, 2.5, 0.1))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_front_wall(self, timeout=4):
        box_name = "front_wall"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0.28
        box_pose.pose.position.y = 0.23
        box_pose.pose.position.z = 0.33
        scene.add_box(box_name, box_pose, size=(0.02, 1.6, 1.46))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_back_wall(self, timeout=4):
        box_name = "back_wall"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = -0.78
        box_pose.pose.position.y = 0.23
        box_pose.pose.position.z = 0.33
        scene.add_box(box_name, box_pose, size=(0.02, 1.6, 1.46))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_ceilling(self, timeout=4):
        box_name = "ceilling"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = -0.32
        box_pose.pose.position.y = 0.23
        box_pose.pose.position.z = 0.85
        scene.add_box(box_name, box_pose, size=(1.2, 1.6, 0.02))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_camera(self, timeout=4):
        box_name = "camera"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = -0.43
        box_pose.pose.position.y = -0.46
        box_pose.pose.position.z = 0.05
        scene.add_box(box_name, box_pose, size=(0.4, 0.15, 0.83))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_base(self, timeout=4):
        box_name = "robot_base"
        scene = self.scene
        ## create a box in the planning scene:
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "base_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 0
        box_pose.pose.position.y = 0
        box_pose.pose.position.z = -0.205
        scene.add_box(box_name, box_pose, size=(0.2, 0.2, 0.41))

        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False)

    def add_objects(self, msg):
        # Add objects for each item in the detection result
        for i in range(len(msg.results.objects)):
            obj_id = msg.results.objects[i].header.frame_id
            # Add objects belong to a same item
            for j in range(len(msg.results.objects[i].poses)):
                obj_name = obj_id+str(j)
                self.add_object(obj_id, obj_name, msg.results.objects[i].poses[j])

    def add_object(self, obj_id, obj_name, pose):
        ''' Add virtual objects in Rviz '''
        # Create an obj pose
        obj_pose = PoseStamped()
        obj_pose.header.frame_id = "base_link"
        obj_pose.pose = pose
        # Update object pose
        obj_pose.pose, obj_ori = self.update_obj_pose(obj_name, obj_pose.pose)
        # Add object to the grasping list if it is selected
        if obj_id in self.order:
            self.calc_grasping_pose(obj_pose.pose, obj_ori)
            self.pickup_list.append(obj_name)
            self.order[obj_id] -= 1
            # Add object to the scene
            scene = self.scene
            size = self.obj_sizes[obj_id]
            scene.add_box(obj_name, obj_pose, size)

        return self.wait_for_state_update(obj_name, box_is_known=True, box_is_attached=False)

    def update_obj_pose(self, obj_name, obj_pose):
        ''' Update object pose '''
        obj_ori = []
        obj_ori.append(obj_pose.orientation.x)
        obj_ori.append(obj_pose.orientation.y)
        obj_ori.append(obj_pose.orientation.z)
        obj_ori.append(obj_pose.orientation.w)
        obj_ori = self.get_ori_on_surface(obj_ori)

        obj_pose.orientation.x = obj_ori[0]
        obj_pose.orientation.y = obj_ori[1]
        obj_pose.orientation.z = obj_ori[2]
        obj_pose.orientation.w = obj_ori[3]

        # Add offset
        obj_pose.position.y += pose_offset_y
        obj_pose.position.z += pose_offset_z

        return obj_pose, obj_ori

    def get_ori_on_surface(self, obj_ori):
        ''' Rotate the detected obj pose to make it perpendicular to the surport surface '''
        '''
        xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        qz = quaternion_about_axis(pi, zaxis)
        obj_ori = quaternion_multiply(qz, obj_ori)
        '''
        euler = list(euler_from_quaternion(obj_ori))
        #print('obj_ori new: ', euler)
        # Set rotation about x and y axis to 0
        euler[0] = 0
        euler[1] = 0
        # Rotate the object orientation if the grasp pose is not reachable
        while abs(euler[2]) > pi/2:
             euler[2] = euler[2] - pi*np.sign(euler[2])
             #print "euler[2] new: %s"%euler[2]
        obj_ori = quaternion_from_euler(euler[0], euler[1], euler[2], 'sxyz')
        return obj_ori

    def calc_grasping_pose(self, obj_pose, obj_ori):
  #      print "obj_ori: %s"%obj_ori
        # select y axis direction as the orientation of a object
        v0 = [0, 1, 0, 0]
        xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)

        # Translation
        rot_mat = quaternion_matrix(obj_ori)
        v1 = np.matmul(rot_mat, v0)   # The orientation of the -y axis regarding of the world coordinate
        trans = v1[0:3]
  #      print trans
        grasp_pos = copy.deepcopy(obj_pose.position)
        grasp_pos.x -= trans[0] * self.grasp_dis
        grasp_pos.y -= trans[1] * self.grasp_dis
        grasp_pos.z -= trans[2] * self.grasp_dis
        pre_grasp_pos = copy.deepcopy(obj_pose.position)
        pre_grasp_pos.x -= trans[0] * self.pre_grasp_dis
        pre_grasp_pos.y -= trans[1] * self.pre_grasp_dis
        pre_grasp_pos.z -= trans[2] * self.pre_grasp_dis

        # Calculate the gripper's pose by rotating from the object position
        grip_rot = [-pi/2, -pi/2, 0]
        qx = quaternion_about_axis(grip_rot[0], xaxis)
        qy = quaternion_about_axis(grip_rot[1], yaxis)
        qz = quaternion_about_axis(grip_rot[2], zaxis)
        grasp_ori = quaternion_multiply(obj_ori,qx)
        grasp_ori = quaternion_multiply(grasp_ori,qy)
  #      print "grasp_ori: %s"%grasp_ori

        # Set preGraspPose
        preGraspPose = copy.deepcopy(obj_pose)
        preGraspPose.position = pre_grasp_pos
        preGraspPose.orientation = Quaternion(grasp_ori[0], grasp_ori[1], grasp_ori[2], grasp_ori[3])
        self.preGraspPoses.append(preGraspPose)

        # Set graspPose
        graspPose = copy.deepcopy(obj_pose)
        graspPose.position = grasp_pos
        graspPose.orientation = Quaternion(grasp_ori[0], grasp_ori[1], grasp_ori[2], grasp_ori[3])
        self.graspPoses.append(graspPose)
        #print "Grasp Pose: %s"%graspPose
        #print "Pre-grasp Pose: %s"%preGraspPose

        self.add_for_grasping(graspPose, trans)

    def add_for_grasping(self, grasp_pose, grasp_ori):
        '''
        Add Grasping pose for the Grasp() function
        TODO: modified the parameters
        '''
        g = moveit_msgs.msg.Grasp()

        # Set grasp pose
        g.grasp_pose.header.frame_id = "base_link"
        g.grasp_pose.pose = grasp_pose

        # Set pre-grasp approach
        g.pre_grasp_approach.direction.header.frame_id = "base_link"
    #    grasp_vector = list(euler_from_quaternion(grasp_ori))
        g.pre_grasp_approach.direction.vector.x = grasp_ori[0]
        g.pre_grasp_approach.direction.vector.y = grasp_ori[1]
        g.pre_grasp_approach.direction.vector.z = grasp_ori[2]
        g.pre_grasp_approach.min_distance = 0.055
        g.pre_grasp_approach.desired_distance = 0.155

        # Set post-grasp retreat
        g.post_grasp_retreat.direction.header.frame_id = "base_link"
        g.post_grasp_retreat.direction.vector.x = 0.0
        g.post_grasp_retreat.direction.vector.y = -1.0
        g.post_grasp_retreat.direction.vector.z = 0.0
        g.post_grasp_retreat.desired_distance = 0.15
        g.post_grasp_retreat.min_distance = 0.05

        # Set posture of eef before grasp
        g.pre_grasp_posture = self.openGripper(g.pre_grasp_posture)

        # Set posture of eef during grasp
        g.grasp_posture = self.closedGripper(g.grasp_posture)

        #g.allowed_touch_objects = [""]
        #g.max_contact_force = 0

        self.grasps.append(g)

    def go_home(self):
        ''' Move the arm to the defined home position in Moveit '''
	group = self.group
	group.set_planning_time(1)
	group.set_num_planning_attempts(6)
	current_state = self.get_robot_state()
	group.set_start_state(current_state)
	group.set_named_target("home")
	plan = group.plan()
	self.execute_plan(plan)

    #    group.clear_pose_targets()
        print "Moved to the home position"

    def go_home_read(self):
        ''' Move the arm to the defined home position in Moveit '''
	if read_traj:
	    plan = self.read_trajectory('go_home.yaml')
            self.execute_plan(plan)
	else:
	    group = self.group
	    #group.set_planning_time(10)
	    #group.set_num_planning_attempts(6)
	    current_state = self.get_robot_state()
	    group.set_start_state(current_state)
	    group.set_named_target("home")
	    plan = group.plan()
	    self.execute_plan(plan)
	    self.save_trajectory(plan, 'go_home.yaml')

    #    group.clear_pose_targets()
        print "Moved to the home position"

    def vac_gripper_control(self, state):
        ''' Vacuum gripper control '''
        fun = 1
        pin = 0
        rospy.wait_for_service('/ur_driver/set_io')
        try:
            gripper_con = rospy.ServiceProxy('/ur_driver/set_io', SetIO)
            gripper_con(fun, pin, state)
    #        print "Set IO succeeded"
            return
        except rospy.ServiceException, e:
            print "Service call for vacuum gripper failed: %s"%e

    def rg2_gripper_control(self, state):
        ''' RG2 gripper control '''
        if flag_sim:
            return
        if io_control:
            fun = 1
            pin = 16

            prefix = "" # TODO:	Modify it to "epicker"

            rospy.wait_for_service(prefix  + '/ur_driver/set_io')
            try:
                gripper_con = rospy.ServiceProxy(prefix  + '/ur_driver/set_io', SetIO)
                gripper_con(fun, pin, state)
                print "Set IO succeeded"
                time.sleep(1.2)
                return
            except rospy.ServiceException, e:
                print "Gripper controll service call failed: %s"%e
        else:
            if state == 1:
                rg2.close_gripper(target_width=10,target_force=30,wait=1.5)
            elif state == 0:
                rg2.open_gripper(target_width=80,target_force=30,wait=1.5)

    def plan_to_pre_pick(self, obj_num):
        '''
        Plan a path to the pre-pick pose
        Todo: remove the obj_num, remove the pose from the list once it is reached
        '''
        group = self.group

        pose_goal = self.preGraspPoses[obj_num]
        #group.set_start_state_to_current_state()
        current_state = self.get_robot_state()
        group.set_start_state(current_state)
        pose_goal_stamped = PoseStamped()
        pose_goal_stamped.pose = pose_goal
        pose_goal_stamped.header.frame_id = "base_link"
        joint_goal = self.get_ik_client(pose_goal_stamped)
	group.set_joint_value_target(joint_goal)
        #group.set_pose_target(pose_goal)
        #print(pose_goal)
        #group.set_planner_id('RRTstarkConfigDefault')
        #group.set_planning_time(6)
        #group.set_num_planning_attempts(6)

        ## Now, we call the planner to compute the plan and execute it.
        #plan = group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        #group.stop()
        plan = group.plan()
        self.execute_plan(plan)
        # It is always good to clear your targets after planning with poses.
        group.clear_pose_targets()

        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def plan_to_pick(self, obj_num, scale=1):
        '''
        Plan a cartesian path to the pick pose
        '''
        group = self.group
        #group.set_start_state_to_current_state()
        current_state = self.get_robot_state()
        group.set_start_state(current_state)
        waypoints = []
        wpose = group.get_current_pose().pose
        waypoints.append(copy.deepcopy(wpose))
        wpose = self.graspPoses[obj_num]
    #    print self.graspPose
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0 disabling:
        (plan, fraction) = group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,        # eef_step
                                           0.0,         # jump_threshold
                                           False)

        self.execute_plan(plan)

    def move_up(self, scale=1):
        '''
        Plan a cartesian path to move up after it grasp the object
        '''
        group = self.group
        #group.set_start_state_to_current_state()
        current_state = self.get_robot_state()
        group.set_start_state(current_state)
        waypoints = []
        wpose = group.get_current_pose().pose
        waypoints.append(copy.deepcopy(wpose))
        #wpose = group.get_current_pose().pose
        wpose.position.z += self.move_up_dis  # Move up
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y -= 0.15  # Move close
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,        # eef_step
                                           0.0,
                                           True)         # jump_threshold

        # Note: We are just planning, not asking move_group to actually move the robot yet:
        self.execute_plan(plan)

    def plan_to_place(self):
        '''
        Plan a path to the place pose
        '''
        group = self.group

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.5
        pose_goal.orientation.x = -0.5
        pose_goal.orientation.y = 0.5
        pose_goal.orientation.z = 0.5
        pose_goal.position.x = 0.05
        pose_goal.position.y = 0.4
        pose_goal.position.z = -0.06
        #group.set_start_state_to_current_state()
        current_state = self.get_robot_state()
        group.set_start_state(current_state)
        pose_goal_stamped = PoseStamped()
        pose_goal_stamped.pose = pose_goal
        pose_goal_stamped.header.frame_id = "base_link"
        joint_goal = self.get_ik_client(pose_goal_stamped)
	    group.set_joint_value_target(joint_goal)
        #group.set_pose_target(pose_goal)
        #group.set_planner_id('RRTstarkConfigDefault')
        #group.set_planning_time(1.5)
        #group.set_num_planning_attempts(6)

        ## Now, we call the planner to compute the plan and execute it.
        #plan = group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        #group.stop()
        plan = group.plan()
        self.execute_plan(plan)
        # It is always good to clear your targets after planning with poses.
        group.clear_pose_targets()

        current_pose = self.group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def openGripper(self, posture):
        '''
        Gripper open function work with the Pick()/Place() function
        Todo: modify the joint names and its parameters for RG2
        '''
        # Add joints of the gripper
        posture.joint_names = ["gripper_joint_r", "gripper_joint_l"]

        # Set them as open
        pos = JointTrajectoryPoint()
        pos.positions.append(0.0)
        pos.positions.append(0.0)
        posture.points.append(pos)
        posture.points[0].time_from_start = rospy.Duration(0.5)

        return posture

    def closedGripper(self, posture):
        '''
        Gripper close function work with the Pick()/Place() function
        Todo: modify the joint names and its parameters for RG2
        '''
        # Add joints of the gripper
        posture.joint_names = ["gripper_joint_r", "gripper_joint_l"]

        # Set them as closed
        pos = JointTrajectoryPoint()
        pos.positions.append(-0.01)
        pos.positions.append(-0.01)
        posture.points.append(pos)
        posture.points[0].time_from_start = rospy.Duration(0.5)

        return posture

    def pick(self, obj_name, grasp):
        ''' The Pick() function '''
        group = self.group
        group.set_num_planning_attempts(5)
        group.allow_replanning(True)

        # Set support surface
        group.set_support_surface_name("bookshelf")

        # Call pick
        group.pick(obj_name, grasp)

    def place(self, obj_name):
        ''' The Place() function '''
        group = self.group
        group.set_num_planning_attempts(5)
        group.allow_replanning(True)
        p = moveit_msgs.msg.PlaceLocation()

        # Set the place pose
        place_pose = geometry_msgs.msg.PoseStamped()
        place_pose.header.frame_id = "base_link"
        place_pose.pose.orientation.w = 0.5
        place_pose.pose.orientation.x = -0.5
        place_pose.pose.orientation.y = 0.5
        place_pose.pose.orientation.z = -0.5
        place_pose.pose.position.x = 0.05
        place_pose.pose.position.y = 0.4
        place_pose.pose.position.z = -0.06

        p.place_pose = place_pose

        # Setting pre-place approach
        p.pre_place_approach.direction.header.frame_id = "base_link"
        p.pre_place_approach.direction.vector.z = -1.0
        p.pre_place_approach.min_distance = 0.03
        p.pre_place_approach.desired_distance = 0.1

        # Setting post-place retreat
        p.post_place_retreat.direction.header.frame_id = "base_link"
        p.post_place_retreat.direction.vector.z = 1.0
        p.post_place_retreat.min_distance = 0.03
        p.post_place_retreat.desired_distance = 0.1

        # Setting posture of eef after placeing object
        p.post_place_posture = self.openGripper(p.post_place_posture)

        # Set support surface
        group.set_support_surface_name("bin")

        # Call place
        group.place(obj_name, p)

    def execute_plan(self, plan):
        ''' Execute a planned path '''
        group = self.group
        group.execute(plan, wait=True)
        group.stop()

    def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        ''' Check out the scene update state '''
        scene = self.scene

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            if is_attached:
              robot_state = self.robot.get_current_state()
              #print "attached object: ", robot_state.attached_collision_objects
              #print("All attached objects: ", scene.get_attached_objects())
              #print("All known objects' name: ", scene.get_known_object_names())
              #print("All known objects ", scene.get_objects())

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                #print "Succeed to update the scene state"
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        print "Failed to update the scene state"
        return False

    def attach_box(self, box_name, timeout=4):
        ''' Attach an object to the end-effector '''
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.group_names

        grasping_group = 'endeffector'
        touch_links = robot.get_link_names(group=grasping_group)
        #print(touch_links)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)

        # Wait for the planning scene to update.
        return self.wait_for_state_update(box_name, box_is_attached=True, box_is_known=False, timeout=timeout)

    def detach_box(self, box_name, timeout=4):
        ''' Detach an object from the end-effector '''
        scene = self.scene
        eef_link = self.eef_link

        scene.remove_attached_object(eef_link, name=box_name)

        # Wait for the planning scene to update.
        return self.wait_for_state_update(box_name, box_is_known=True, box_is_attached=False, timeout=timeout)

    def remove_box(self, box_name, timeout=4):
        ''' Remove an object from the scene '''
        scene = self.scene
        scene.remove_world_object(box_name)

        return self.wait_for_state_update(box_name, box_is_attached=False, box_is_known=False, timeout=timeout)

    def get_obj_poses_client(self):
        ''' Get poses of all kind of objects from object detection system '''
        rospy.wait_for_service('target_poses')
        try:
            obj_poses = rospy.ServiceProxy('target_poses', TargetPoses)
            res = obj_poses("get detection results") # Todo: might delete the input
  #          print 'result: %s'%res.data
            self.add_objects(res)
        except rospy.ServiceException, e:
            print "Service call for getting objects pose failed: %s"%e

    def print_joint_val(self):
        current_joints = self.group.get_current_joint_values()
        print("current joint value: ", current_joints)

    def get_robot_state(self):
        ''' Get robot state from the planning scene '''
        rospy.wait_for_service('/get_planning_scene')
        components = moveit_msgs.msg.PlanningSceneComponents()
        components.components = 4
        try:
            planning_scene_proxy = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene())
            planning_scene = planning_scene_proxy(components)
            #print 'Got robot state: %s'%planning_scene.scene.robot_state
            return planning_scene.scene.robot_state
        except rospy.ServiceException, e:
            print "Service call for getting robot state failed: %s"%e

    def get_ik_client(self, pose):
        group = self.group
        ik_rqt = moveit_msgs.msg.PositionIKRequest()
        ik_rqt.group_name = "manipulator"#self.group_names
        ik_rqt.robot_state = self.robot.get_current_state()
	ik_rqt.ik_link_name = "ee_link"
	ik_rqt.pose_stamped = pose
	#ik_rqt.timeout = 5.0
	ik_rqt.attempts = 8

        rospy.wait_for_service('compute_ik')
        try:
            get_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
            robot_state = get_ik(ik_rqt)
            return robot_state.solution.joint_state
        except rospy.ServiceException, e:
            print "Get IK service call failed: %s"%e

    def save_trajectory(self, plan, file_name):
        file_path = traj_path + file_name
        with open(file_path, 'w') as file_save:
            yaml.dump(plan, file_save, default_flow_style=True)

    def read_trajectory(self, file_name):
        file_path = traj_path + file_name
        with open(file_path, 'r') as file_open:
            loaded_plan = yaml.load(file_open)
        return loaded_plan


def main():
    global rg2
    global pose_offset_x
    global pose_offset_y
    global pose_offset_z
    global params

    # To indicate it is simulation or not
    if flag_sim:
        print "simulation"

    # Load parameters
    if len(sys.argv) > 2:
        config_name = sys.argv[2]
    else:
        config_name = "config_kinect.yaml"
    params = None
    yaml_path = '../config/{}'.format(config_name)
    print("yaml_path: ", yaml_path)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

    pose_offset_x = params["pose_offset"]['x']
    pose_offset_y = params["pose_offset"]['y']
    pose_offset_z = params["pose_offset"]['z']

    # Publish state to stop object detection
    pub_state = rospy.Publisher('detection_state', Bool, queue_size=10)
    try:
        # Initialization
        emo = PickNPlace()
        urscript = URScript("/ur_driver/URScript") # TODO: Modify it to "/epicker/..."
        rg2 = onrobot_rg2.OnRobotGripperRG2(urscript)

        # Load environment
        #time.sleep(1)
        emo.load_environment()
        print "Loaded the virtual environment"

        # Move the arm to the home position
        emo.go_home() # if the moveit home pose doen't work uncomment this line
        time.sleep(1)

        # Order
        print "===>Please enter the quantity of the JDB you would like to order. Press `Enter` to confirm ..."
        emo.order['JDB'] = int(raw_input())
        print "===>Please enter the quantity of the YZ you would like to order. Press `Enter` to confirm ..."
        emo.order['YZ'] = int(raw_input())
        print "Going to pick %s JDB and %s YZ." % (emo.order['JDB'], emo.order['YZ'])

        # Get the detected objects pose
        emo.get_obj_poses_client()
        #rospy.sleep(0.5)
        time.sleep(0.5)

        # Stop object detection  #TODO: make it a function stop_detction()
        msg = Bool()
        msg.data = False
        pub_state.publish(msg)

        # Pick the selected objects
        for i in range(len(emo.pickup_list)):
            obj_name = emo.pickup_list[i]
            print "Going to pick %s"%obj_name

            # Picking
            if flag_pp:
                emo.pick(obj_name, emo.grasps[i])
            else:
                emo.plan_to_pre_pick(i)
                #print "Planned a path for picking the object"
                time.sleep(0.5)
                emo.plan_to_pick(i)
                if not flag_sim:
                    # close the gripper
                    emo.rg2_gripper_control(True)
                    time.sleep(1.0)
            print "Finished the picking motion"

            # Attach the object
            if not flag_pp:  # Todo: may merge to the picking process
                emo.attach_box(obj_name)
                #time.sleep(3)
                #emo.group.attach_object(obj_name, "ee_link")
                #print "Object attached"

            # Placing
            if flag_pp:
                rospy.sleep(0.5)
                emo.place(obj_name)
            else:
                emo.move_up()
                time.sleep(0.5)
                emo.plan_to_place()
                #emo.print_joint_val()
                # Open the gripper
                if not flag_sim:
                    emo.rg2_gripper_control(False)
                    time.sleep(1.0)
                # Detach the object
                #emo.group.detach_object(obj_name)
                emo.detach_box(obj_name)
            print "Finished the placing motion"

            # Move back to the home position
            emo.go_home_read()

            # Remove the object
            #emo.detach_box(obj_name)
            emo.remove_box(obj_name)
            #print "Object removed"
            time.sleep(0.5)
        print "============ Epicker pick&place demo complete!"
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()
