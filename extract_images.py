import h5py
import numpy as np
import cv2

def load_gt4_file(file_path):
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data_dict[key] = np.array(f[key])
    return data_dict

FILE_TEMPLATE = "/home/ubuntu/multiplayer-racing-low-res/dataset_multiplayer_racing_{}.hdf5"
frame_x_dict = {}
frame_y_dict = {}
for i in range(1, 2):
    try:
        file_path = FILE_TEMPLATE.format(i)
        data_dict = load_gt4_file(file_path)
        frame_x_dict.update({f"{i}_{k}": v for k, v in data_dict.items() if "_x" in k})
        frame_y_dict.update({f"{i}_{k}": v for k, v in data_dict.items() if "_y" in k})
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        continue

for i, value in enumerate(frame_x_dict.values()):
    first = value[:,:,:3]
    cv2.imwrite(f"images/{i}.png", first)
    
