import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import cv2

# Load the pre-trained Faster R-CNN model
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Define the COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold=0.5):
    # Transform the image
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img.convert("RGB"))
    pred = model([img])
    pred = pred[0]
    
    # Process predictions
    pred_score = pred['scores'].detach().numpy()
    high_scores_indices = np.where(pred_score > threshold)[0]
    
    if len(high_scores_indices) == 0:
        return [], []

    high_scores_indices = high_scores_indices[-1] + 1

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in pred['labels'][:high_scores_indices].numpy()]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in pred['boxes'][:high_scores_indices].detach().numpy()]

    return pred_boxes, pred_class

def draw_boxes(boxes, classes, img):
    img_np = np.array(img)
    
    for i in range(len(boxes)):
        box = boxes[i]
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[1][0]), int(box[1][1]))
        img_np = cv2.rectangle(img_np, pt1, pt2, color=(0, 255, 0), thickness=2)
        text_origin = (pt1[0], pt1[1] - 10)
        img_np = cv2.putText(img_np, classes[i], text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
    
    return img_np

def main():
    st.title("Object Finder 🔍")

    st.markdown("""
    This is an object detector model. Use images containing various objects or tools for the best results 🙂
    """)

    uploaded_image = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        width, height = image.size
        st.write(f"Image Dimensions: {width}x{height}")

        if st.button("Identify the objects"):
            st.info("Detecting objects...")

            boxes, pred_cls = get_prediction(image, 0.5)
            
            if pred_cls:
                st.write("The objects detected are:")
                st.write(", ".join(sorted(pred_cls)))
            else:
                st.warning("No objects detected. Try uploading a different image.")

            detected_image = draw_boxes(boxes, pred_cls, image)
            st.image(detected_image, caption='Detected Image.', use_column_width=True)

            st.success("Detection completed successfully!")

if __name__ == "__main__":
    main()
