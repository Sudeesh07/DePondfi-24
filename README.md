# DePondfi-24

# Steps to run the YOLO_train.ipynb script

We kindly request you to use a markdown compatible app such as VS Code or an online website for viewing this file.

## Requirements

A python version 3.9 or above is required to run this script.

This script requires the following libraries to be installed:
- OpenCV
- Numpy
- OS
- Ultralytics
- Scikit-learn


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install opencv-python numpy os ultralytics scikit-learn
```

## Steps

1. In the first cell edit the path of input_folder according to your folder setup. Similarity update the output_folder according to your folder structure
```python
#preprocess the images 
import os
import cv2
import numpy as np

def preprocess_underwater_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge([enhanced_l_channel, a_channel, b_channel])
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    gamma = 1.5
    enhanced_image = np.power(enhanced_image / 255.0, gamma)
    enhanced_image = np.uint8(enhanced_image * 255)

    return enhanced_image

def preprocess_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(input_path)
            preprocessed_image = preprocess_underwater_image(image)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, preprocessed_image)
            print(f"Processed and saved: {output_path}")

# Define input and output folders
input_folder = r'path_to\DePondFi24_Train\DePoondFi24_Train\Images'
output_folder = r'path_to\preprocessed_images'

preprocess_images_in_folder(input_folder, output_folder)

```

2. Similarly, change the folder paths of the ***keypoint_folder*** and ***output_folder*** in the 2nd cell 

```python
#converting anotations to yolo format
# Set the paths for the image and keypoint folders
keypoint_folder = 'path_to\DePondFi24_Train\DePoondFi24_Train\Keypoints'

# Set the output folder for the YOLO annotations
output_folder = 'path_to\YOLO_annotations_updated'
height = 640
width = 640
os.makedirs(output_folder, exist_ok=True)
print("Created the output folder")
# Loop through all images in the image folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        base_name = os.path.splitext(filename)[0]
        keypoint_file = os.path.join(keypoint_folder, f'{base_name}.txt')

        keypoints = []
        with open(keypoint_file, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split(','))
                keypoints.append((x, y))

        output_file = os.path.join(output_folder, f'{base_name}.txt')
        with open(output_file, 'w') as f:
            for i in range(0, len(keypoints), 9):
                # Extract the x and y coordinates
                x_coords = [x for x, y in keypoints[i:i+9]]
                y_coords = [y for x, y in keypoints[i:i+9]]
                # Calculate the bounding box coordinates
                fish_center_x, fish_center_y = x_coords[3] , y_coords[3]
                # Calculate the center of the bounding box
                x_center = fish_center_x / width
                y_center = fish_center_y / height
                bbw = (max(x_coords) - min(x_coords)) / width
                bbh = (max(y_coords) - min(y_coords)) / height
                # Calculate the width and height of the bounding box
                # Write the annotation to the file
                mouth_x , mouth_y = x_coords[0]/width, y_coords[0]/height
                eye_x, eye_y = x_coords[1]/width, y_coords[1]/height
                top_fin_x, top_fin_y = x_coords[2]/width, y_coords[2]/height
                fish_center_x , fish_center_y = fish_center_x/width, fish_center_y/height
                bottom_center_x , bottom_center_y = x_coords[4]/width , y_coords[4]/height
                start_point_tail_x, start_point_tail_y = x_coords[5]/width, y_coords[5]/height
                top_outline_tail_x , top_outline_tail_y = x_coords[6]/width, y_coords[6]/height
                mid_outline_tail_x , mid_outline_tail_y = x_coords[7]/width, y_coords[7]/height
                bottom_outline_tail_x , bottom_outline_tail_y = x_coords[8]/width, y_coords[8]/height
                f.write(f'0 {x_center} {y_center} {bbw} {bbh} {mouth_x} {mouth_y} {eye_x} {eye_y} {top_fin_x} {top_fin_y} {fish_center_x} {fish_center_y} {bottom_center_x} {bottom_center_y} {start_point_tail_x} {start_point_tail_y} {top_outline_tail_x} {top_outline_tail_y} {mid_outline_tail_x} {mid_outline_tail_y} {bottom_outline_tail_x} {bottom_outline_tail_y}\n')

```

3. In the 3rd cell, make sure to do the same and update the paths that were set above with respect to your folder structure. 
    
```python
#dividing the images into train and test for YOlO 
import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = 'path_to\preprocessed_images'
annotations_dir = 'path_to\YOLO_annotations_updated'
output_dir = 'path_to\YOLO_training/'

# Create output directories
train_images_dir = os.path.join(output_dir, 'images', 'train')
train_labels_dir = os.path.join(output_dir, 'labels', 'train')
val_images_dir = os.path.join(output_dir, 'images', 'val')
val_labels_dir = os.path.join(output_dir, 'labels', 'val')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split the dataset into train and test
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

def copy_files(files, src_dir, dst_dir, extension):
    for file in files:
        name, _ = os.path.splitext(file)
        src_path = os.path.join(src_dir, f"{name}{extension}")
        dst_path = os.path.join(dst_dir, f"{name}{extension}")
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

copy_files(train_files, images_dir, train_images_dir, '.jpg')
copy_files(train_files, annotations_dir, train_labels_dir, '.txt')

copy_files(test_files, images_dir,val_images_dir, '.jpg')
copy_files(test_files, annotations_dir, val_labels_dir, '.txt')

print("Dataset split and copied successfully.")
```
4. Create a data.yaml file in the folder with the images, for assigning the path of the train and val directories 
The YAML file should be like 
```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: YOLO_training # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images

# Keypoints
kpt_shape: [9, 2]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)

# Classes dictionary
names:
  0: fish
```

6. For training a YOLO model, a strict folder structure needs to be followed. 
```bash
-Input_folder
-Preprocessed_images_directory
-YOLO_train
	-images
		-train
			-image1.jpg
			-image2.jpg
			……
		-val
			-image4.jpg
			………
	-labels
		–train
			-image1.txt
			-image2.txt
			-val
				-image4.txt
				…………
    -data.yaml
```

7. After the creating of the YAML file and setting up the folder for training YOLO, ensure that in the settings.yaml file is created and datasets_dir is set to the directory containing the YOLO_training folder.
The location of this file varies depending on the installation method, but it is typically located inside ***AppData/Roaming/Roboflow/settings.yaml*** on Windows or ***~/.roboflow/settings.yaml*** on Linux or MacOS.
The settings.yaml file should look like this:
```yaml
settings_version: 0.0.4
datasets_dir: C:\Users\sudee\Documents\Projects\Pondify\
weights_dir: weights
runs_dir: runs
uuid: 58e20ffefdb998c50daf45b617ed73a109b955824034c8a8f69b93f18beaae0b
sync: true
api_key: ''
openai_api_key: ''
clearml: true
comet: true
dvc: true
hub: true
mlflow: true
neptune: true
raytune: true
tensorboard: true
wandb: true
```

8. After updating the settings.yaml file run the next cell, this will initiate the training of a YOLO model. Adjust the batch_size according to your system specifications
(less batch_size for less compute power )
```python
import torch
from ultralytics import YOLO

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights
#change the paths accordingly
results = model.train(data="YOLO_training\data.yaml", epochs=100, imgsz=640,device=device,batch = 80, plots = True, box = 10.0, pose = 15.0, cos_lr = True, patience = 10, project = "Pondify", name = "yolov8n-pose_test_3", exist_ok = True)
```

9. After the training is completed, the model will be saved in the weights folder. The model can be used for inference on new images.
    
```python
import os 
import cv2

#traversing through the images and predicting the bounding boxes

# Load the model
#change the paths accordingly
model = YOLO(r"Pondify\yolov8n-pose_test_3\weights\last.pt")  # load a custom model

images_dir = r"preprocessed_images"
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

os.makedirs("results", exist_ok=True)

for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)

    results = model(image_path)

    output_path = os.path.join("results", image_file)

    for result in results:
        result.save(filename=output_path)   

    print(f"Processed and saved: {output_path}")


    output_path = os.path.join("keypoint_predictions", f"{os.path.splitext(image_file)[0]}.txt")
    with open(output_path, "w") as file:
        for result in results:
            keypoints = result.keypoints  
            for i in range(keypoints.xy.shape[0]):
                for j in range(keypoints.xy.shape[1]):
                    x = keypoints.xy[i][j][0]
                    y = keypoints.xy[i][j][1]
                    file.write(f"{float(x)},{float(y)}\n")
        print(f"Processed and saved: {output_path}")
```



## Results
After running this notebook, we will be getting a folder that contains the weights of the model and the predictions of the bounding boxes and keypoints on the images.
The keypoints are present in the folder in the form of text files with the same name as the image file. 

![alt text](image.png)

## Contributors
- Sudeesh Muthu. M 
- Sai Eswara Murali
- Afzal Ahmed
- Pratyush. N.K
