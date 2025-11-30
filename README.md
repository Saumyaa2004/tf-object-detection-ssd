Sign Language Detection (TensorFlow SSD)

A sign language detection project built using TensorFlow’s SSD MobileNet model.
The pipeline includes dataset collection, TFRecord generation, model training, and live inference.

Features

Detection using SSD MobileNet

Custom dataset support

Trained using TensorFlow Object Detection API

Works with webcam or pre-recorded video

Tech Stack

Python

TensorFlow 2.x

TensorFlow Object Detection API

OpenCV

How to Run

Install dependencies:

pip install -r requirements.txt


Run training:

python model_main_tf2.py --model_dir=workspace/models/my_ssd_mobnet --pipeline_config_path=workspace/models/my_ssd_mobnet/pipeline.config


Run detection script:

python detect.py

Dataset

Includes custom-labeled images for sign language gestures, split into train and test.

Output

The model generates:

Checkpoints

Exported inference graph

Real-time detection window
