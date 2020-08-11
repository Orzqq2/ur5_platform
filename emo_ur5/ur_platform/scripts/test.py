import matplotlib.pyplot as plt
import time
import h5py

demo_file = h5py.File('storage_date_100_130_fast_2.hdf5', 'r')
a = demo_file['storage_robot_state_image0']
b = demo_file['storage_object_state0']
print len(b)
for i in range(50, len(a)):
    print i
    # plt.imshow(a[i, :, :, 2])
    # plt.show()
    plt.imshow(a[i, :, :, 3])
    plt.show()
    print b[i]
    time.sleep(0.5)
# plt.imshow(a[58, :, :, 3])
# plt.show()