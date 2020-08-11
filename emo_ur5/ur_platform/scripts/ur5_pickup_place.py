#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

## BEGIN_SUB_TUTORIAL imports
##
## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. More on these below. We also import `rospy`_ and some messages that we will use:
##

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
import yaml
from math import pi
from std_msgs.msg import String
from moveit_msgs.srv import GetPlanningScene, GetPositionIK
from geometry_msgs.msg import PoseArray, Quaternion, PoseStamped
from ur_msgs.srv import SetIO
from tf.transformations import *
from math import pi
from std_msgs.msg import String, Float64, Bool
from moveit_commander.conversions import pose_to_list
from trajectory_msgs.msg import JointTrajectoryPoint

## END_SUB_TUTORIAL
script_dir = os.path.dirname(__file__)
platform_model_rel_path = "../model/"
platform_model_path = os.path.join(script_dir, platform_model_rel_path)


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


class UR5PickupPlace(object):
    """MoveGroupPythonIntefaceTutorial"""

    def __init__(self):
        super(UR5PickupPlace, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('UR5_pickup_place', anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "manipulator"
        move_group_arm = moveit_commander.MoveGroupCommander(group_name)

        group_name = "gripper"
        move_group_gripper = moveit_commander.MoveGroupCommander(group_name)

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group_arm.get_planning_frame()
        # print "============ Planning frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group_arm.get_end_effector_link()
        # print "============ End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        # print "============ Available Planning Groups:", robot.get_group_names()

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        # print "============ Printing robot state"
        # print robot.get_current_state()
        # print ""
        ## END_SUB_TUTORIAL

        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group_arm = move_group_arm
        self.move_group_gripper = move_group_gripper
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def gipper_control(self, state='open'):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group_gripper = self.move_group_gripper

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_ so the first
        ## thing we want to do is move it to a slightly better configuration.
        # We can get the joint values from the group and adjust some of the values:
        joint_goal = move_group_gripper.get_current_joint_values()
        if state == 'open':
            joint_goal[2] = 0.0
        else:
            joint_goal[2] = 0.6

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group_gripper.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group_gripper.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group_gripper.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self, pose_goal):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group_arm = self.move_group_arm

        move_group_arm.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group_arm.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group_arm.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group_arm.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        # current_pose = self.move_group_arm.get_current_pose().pose
        # return all_close(pose_goal, current_pose, 0.01)

    def plan_cartesian_path(self, scale=1):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group_arm = self.move_group_arm

        ## BEGIN_SUB_TUTORIAL plan_cartesian_path
        ##
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        ##
        waypoints = []

        wpose = move_group_arm.get_current_pose().pose
        wpose.position.z += scale * 0.1  # First move up (z)
        wpose.position.y += scale * 0.2  # and sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.x -= scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y -= scale * 0.1  # Third move sideways (y)
        waypoints.append(copy.deepcopy(wpose))

        # We want the Cartesian path to be interpolated at a resolution of 1 cm
        # which is why we will specify 0.01 as the eef_step in Cartesian
        # translation.  We will disable the jump threshold by setting it to 0.0,
        # ignoring the check for infeasible jumps in joint space, which is sufficient
        # for this tutorial.
        move_group_arm.set_start_state_to_current_state()
        plan, fraction = move_group_arm.compute_cartesian_path(waypoints, 0.01, 0.0, True)
        # Note: We are just planning, not asking move_group to actually move the robot yet:
        return plan, fraction

    def execute_plan(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group_arm = self.move_group_arm

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group_arm.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL

    def wait_for_state_update(self, box_name, box_is_known=False, box_is_attached=False, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Receieved
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL

    def add_ground(self, timeout=4):
        scene = self.scene
        rospy.sleep(2)
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = 0
        platform_pose.pose.position.y = 0
        platform_pose.pose.position.z = -0.40
        object_name = 'ground'
        scene.add_box(object_name, platform_pose, size=(10, 10, 0.001))
        return self.wait_for_state_update(object_name, box_is_known=True, box_is_attached=False)

    def add_wall1(self, timeout=4):
        scene = self.scene
        rospy.sleep(1)
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = -0.65
        platform_pose.pose.position.y = 0
        platform_pose.pose.position.z = -0.30
        object_name = 'wall1'
        scene.add_box(object_name, platform_pose, size=(0.01, 10, 2))
        return self.wait_for_state_update(object_name, box_is_known=True, box_is_attached=False)

    def add_wall2(self, timeout=4):
        scene = self.scene
        rospy.sleep(1)
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = 0
        platform_pose.pose.position.y = -0.55
        platform_pose.pose.position.z = -0.30
        object_name = 'wall2'
        scene.add_box(object_name, platform_pose, size=(10, 0.01, 2))
        return self.wait_for_state_update(object_name, box_is_known=True, box_is_attached=False)

    def add_ceiling(self, timeout=4):
        scene = self.scene
        rospy.sleep(1)
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = 0
        platform_pose.pose.position.y = 0
        platform_pose.pose.position.z = 0.65
        object_name = 'ceiling'
        scene.add_box(object_name, platform_pose, size=(2, 2, 0.01))
        return self.wait_for_state_update(object_name, box_is_known=True, box_is_attached=False)

    def add_object(self, object_name, timeout=4):
        scene = self.scene
        rospy.sleep(2)
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = 0.65
        platform_pose.pose.position.y = 0.5
        platform_pose.pose.position.z = 0.09
        mesh_name = object_name
        scene.add_box(object_name, platform_pose, size=(0.03, 0.03, 0.1))
        return self.wait_for_state_update(mesh_name, box_is_known=True, box_is_attached=False)

    def remove_box(self, box_name, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL remove_object
        ##
        ## Removing Objects from the Planning Scene
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can remove the box from the world.
        scene.remove_world_object(box_name)

        ## **Note:** The object must be detached before we can remove it from the world
        ## END_SUB_TUTORIAL

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_name, box_is_attached=False, box_is_known=False, timeout=timeout)

    def plan_to_pre_pick(self, obj_num):
        '''
        Plan a path to the pre-pick pose
        Todo: remove the obj_num, remove the pose from the list once it is reached
        '''
        move_group_arm = self.move_group_arm

        # group.set_start_state_to_current_state()
        platform_pose = geometry_msgs.msg.PoseStamped()
        platform_pose.header.frame_id = "base_link"
        platform_pose.pose.orientation.w = 1.0
        platform_pose.pose.position.x = 0.4
        platform_pose.pose.position.y = 0.5
        platform_pose.pose.position.z = 0.09
        move_group_arm.set_start_state_to_current_state()
        self.go_to_pose_goal(platform_pose)

        # current_pose = move_group_arm.get_current_pose().pose
        # return all_close(platform_pose, current_pose, 0.01)

    def plan_to_pick(self, obj_num, scale=1):
        '''
        Plan a cartesian path to the pick pose
        '''
        move_group_arm = self.move_group_arm
        waypoints = []
        wpose = move_group_arm.get_current_pose().pose
        wpose.position.x += 0.11  # Move close
        waypoints.append(copy.deepcopy(wpose))
        plan, fraction = move_group_arm.compute_cartesian_path(waypoints, 0.01, 0.0, True)
        self.execute_plan(plan)

    def move_up(self):
        move_group_arm = self.move_group_arm
        # move_group_arm.set_start_state_to_current_state()
        waypoints = []
        wpose = move_group_arm.get_current_pose().pose
        wpose.position.z += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.x -= 0.15  # Move close
        waypoints.append(copy.deepcopy(wpose))
        plan, fraction = move_group_arm.compute_cartesian_path(waypoints, 0.01, 0.0, True)
        self.execute_plan(plan)

    def plan_to_place(self):
        '''
        Plan a path to the place pose
        '''
        move_group_arm = self.move_group_arm

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.5
        pose_goal.orientation.x = -0.5
        pose_goal.orientation.y = 0.5
        pose_goal.orientation.z = 0.5
        pose_goal.position.x = 0.6
        pose_goal.position.y = -0.25
        pose_goal.position.z = 0
        # group.set_start_state_to_current_state()
        self.go_to_pose_goal(pose_goal)

    def attach_box(self, box_name, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        scene = self.scene
        eef_link = self.eef_link
        group_names = self.move_group_arm

        grasping_group = "gripper"
        touch_links = robot.get_link_names(group=grasping_group)
        scene.attach_box(eef_link, box_name, touch_links=touch_links)
        self.gipper_control('close')
        ## END_SUB_TUTORIAL

    def detach_box(self, box_name, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        scene = self.scene
        eef_link = self.eef_link

        ## BEGIN_SUB_TUTORIAL detach_object
        ##
        ## Detaching Objects from the Robot
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## We can also detach and remove the object from the planning scene:
        scene.remove_attached_object(eef_link, name=box_name)
        self.gipper_control('open')


def main():
    try:
        ## Initialize UR5 pickup and place
        tutorial = UR5PickupPlace()
        tutorial.add_ground()
        tutorial.add_wall1()
        tutorial.add_wall2()
        tutorial.add_ceiling()

        ## test plan and pickup
        # plan, _ = tutorial.plan_cartesian_path(1)
        # tutorial.execute_plan(plan)
        # tutorial.gipper_control('open')

        ## test add object and plan
        tutorial.add_object('bin')
        rospy.sleep(2)
        # tutorial.remove_box("bin")
        tutorial.plan_to_pre_pick(1)
        tutorial.plan_to_pick(1)
        tutorial.attach_box('bin')
        tutorial.move_up()
        tutorial.plan_to_place()
        tutorial.detach_box('bin')



    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
