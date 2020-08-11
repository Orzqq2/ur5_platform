import matplotlib.pyplot as plt
import time
import h5py

demo_file = h5py.File('data/storage_date.hdf5', 'r')
a = demo_file['storage_robot_state_image5']
for i in range(len(a)):
    print i
    plt.imshow(a[i, :, :, 1])
    plt.show()
    time.sleep(0.3)
# plt.imshow(a[60, :, :, 2])
# plt.show()
