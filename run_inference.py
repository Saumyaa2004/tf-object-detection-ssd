import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image

# ---------- USER SETTINGS ----------
CONFIG_PATH = r"C:\RealTimeObjectDetection\Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config"
CHECKPOINT_PATH = r"C:\RealTimeObjectDetection\Tensorflow\workspace\models\my_ssd_mobnet"
LABEL_MAP_PATH = r"C:\RealTimeObjectDetection\Tensorflow\workspace\annotations\label_map.pbtxt"
IMAGE_DIR = r"C:\RealTimeObjectDetection\Tensorflow\workspace\images\test"
OUTPUT_DIR = r"C:\RealTimeObjectDetection\Tensorflow\workspace\images\output"
# -----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pipeline config and build model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint (latest)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(tf.train.latest_checkpoint(CHECKPOINT_PATH)).expect_partial()

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Run inference on all images in IMAGE_DIR
for img_file in os.listdir(IMAGE_DIR):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_file)
    image = np.array(Image.open(img_path))
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # Extract detection data
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.3,
        agnostic_mode=False
    )

    # Save output
    output_path = os.path.join(OUTPUT_DIR, img_file)
    Image.fromarray(image).save(output_path)
    print(f"Saved: {output_path}")

print("Inference complete for all images.")
