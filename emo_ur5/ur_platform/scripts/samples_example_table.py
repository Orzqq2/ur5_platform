import random
import rospy
import time
# from receive_message import ReceiveROSMessage
from ur5_table_pickup_place_for_sample import UR5PickupPlace
from gazebo_msgs.msg import ModelState, ModelStates

# Initializer
tutorial = UR5PickupPlace()
tutorial.add_ground()
tutorial.add_wall1()
tutorial.add_wall_left()
tutorial.add_wall2()
tutorial.add_ceiling()

record_success = 0
record_fauiler = 0


def write_file(txt='string'):
    f = open('tmp.txt', 'w')
    f.write(txt)
    f.close()


def read_file_record():
    f = open('record_state.txt')
    return f.readline()


def write_file_record(txt='string'):
    f = open('record_state.txt', 'w')
    f.write(txt)
    f.close()


write_file_record('1')
for i in range(600):
    print i
    print record_success
    # Set random target location x, y
    while read_file_record() == 'writing':
        time.sleep(0.05)
        pass
    object_x = random.uniform(-0.59, -0.45)
    object_y = random.uniform(0.5, 0.7)
    # Add object
    tutorial.add_object(object_x, object_y, object_name='bin')
    # Add object in Gazebo
    tutorial.gazebo_reset_object()
    tutorial.gazebo_set_object(object_x, object_y + 0.2)
    # rospy.sleep(0.5)
    # tutorial.move_up_slight()
    rospy.sleep(0.5)
    tutorial.arm_init_before_pickup()
    rospy.sleep(0.5)
    write_file('Start')
    tutorial.plan_to_pre_pick(object_x + 0.2, object_y)
    rospy.sleep(0.5)
    tutorial.plan_to_pick(1)
    tutorial.attach_box('bin')
    rospy.sleep(0.5)
    tutorial.move_up()
    rospy.sleep(0.5)
    tutorial.plan_to_place()
    tutorial.detach_box('bin')

    # Record result
    get_object_x, get_object_y = tutorial.get_object_position()
    if 0.4 < get_object_y - 0.2 < 0.6 and 0.3 < get_object_x < 0.54:
        record_success += 1
    else:
        record_fauiler += 1
    write_file('Finish')
    tutorial.go_home()
    #
    # if i % 3 == 2:
    #     rospy.wait_for_service('/gazebo/reset_simulation')
    #     print 'reset gazebo'
