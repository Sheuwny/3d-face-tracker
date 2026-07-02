import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
import face_alignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device=device)

def detect_3d_face_analytics(input_image):
    if input_image is None: return None, "Please upload an image."
    image_np = np.array(input_image)
    annotated_image = image_np.copy()
    preds = fa.get_landmarks(image_np)
    if preds is None or len(preds) == 0: return input_image, "No face found."
    
    landmarks_3d = preds[0]
    for point in landmarks_3d:
        cv2.circle(annotated_image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    return Image.fromarray(annotated_image), "Pipeline Status: SUCCESS"

demo = gr.Interface(fn=detect_3d_face_analytics, inputs=gr.Image(type="pil"), outputs=[gr.Image(type="pil"), gr.Textbox()])
if __name__ == "__main__":
    demo.launch(share=True)
