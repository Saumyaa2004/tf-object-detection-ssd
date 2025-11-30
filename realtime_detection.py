import os
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder_tf2 as model_builder

# -----------------------
# PATHS - UPDATE THESE
# -----------------------
PIPELINE_CONFIG = "C:/RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config"
CHECKPOINT_DIR = "C:/RealTimeObjectDetection/Tensorflow/workspace/models/my_ssd_mobnet/"
LABEL_MAP_PATH = "C:/RealTimeObjectDetection/Tensorflow/workspace/annotations/label_map.pbtxt"

# -----------------------
# LOAD MODEL
# -----------------------
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
ckpt.restore(latest_checkpoint).expect_partial()

# -----------------------
# LOAD LABEL MAP
# -----------------------
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# -----------------------
# DETECTION FUNCTION
# -----------------------
@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# -----------------------
# REAL-TIME WEBCAM
# -----------------------
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # Visualization
    from object_detection.utils import visualization_utils as viz_utils

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5,
        line_thickness=3
    )

    cv2.imshow('Real-Time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
