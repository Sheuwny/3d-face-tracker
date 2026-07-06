import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, pipeline

# MediaPipe standard modules
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

device = "cpu"

print("Loading Models on CPU... ")
# 1. BLIP VQA Model
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# 2. Object Detector
obj_detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# MediaPipe Engines Setup
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
pose_tracker = mp_pose.Pose(static_image_mode=True, model_complexity=1)

# CUSTOM DARK COLOURED DRAWING SPECS (OpenCV uses BGR format)
# Face ke liye Dark Slate/Gray lines
dark_face_lines = mp_drawing.DrawingSpec(color=(60, 60, 60), thickness=1, circle_radius=1)

# Pose ke liye Thicker Dark Navy Blue lines aur Dark Red joints
dark_pose_joints = mp_drawing.DrawingSpec(color=(0, 0, 139), thickness=4, circle_radius=4) # Dark Red
dark_pose_lines = mp_drawing.DrawingSpec(color=(139, 0, 0), thickness=3) # Dark Blue

# Specific Face Parts Tracker (Deep Bold Colors)
COLORS = {
    "left eye": (139, 0, 0),    # Dark Blue
    "right eye": (102, 0, 102), # Dark Purple
    "nose": (0, 100, 0),        # Dark Green
    "mouth": (0, 0, 139)        # Dark Red
}

LANDMARK_GROUPS = {
    "left eye": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "right eye": [362, 382, 381, 380, 374, 373, 390, 249],
    "nose": [1, 2, 3, 4, 168, 197],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405]
}

def process_ultimate_grounding(image, question, target_face_parts):
    if image is None:
        return "Image upload karo!", "N/A", "N/A", "N/A", None

    # --- Part 1: AI Chat (BLIP VQA) ---
    inputs = blip_proc(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    ai_answer = blip_proc.decode(out[0], skip_special_tokens=True)

    # Convert Image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # --- Part 2: 3D Face Mesh (Dark Styled) ---
    face_results = face_mesh.process(img_rgb)
    face_coords = "No Face Detected"
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # Draw face wireframe with Custom Dark Gray lines
        mp_drawing.draw_landmarks(
            image=img_cv, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None, connection_drawing_spec=dark_face_lines
        )
        
        face_coords = ""
        requested_parts = [p.strip().lower() for p in target_face_parts.split(",") if p.strip()]
        for part in requested_parts:
            if part in LANDMARK_GROUPS:
                indices = LANDMARK_GROUPS[part]
                color = COLORS[part]
                xs = [face_landmarks.landmark[idx].x for idx in indices]
                ys = [face_landmarks.landmark[idx].y for idx in indices]
                zs = [face_landmarks.landmark[idx].z for idx in indices]
                
                # Highlight key face features with thick dark dots
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    cv2.circle(img_cv, (int(lm.x*w), int(lm.y*h)), 3, color, -1)
                
                fx = (sum(xs)/len(xs) - 0.5) * 200
                fy = (sum(ys)/len(ys) - 0.5) * 200
                fz = (sum(zs)/len(zs)) * 100
                face_coords += f"📍 {part.capitalize()}: X={fx:.1f}, Y={fy:.1f}, Z={fz:.1f}\n"

    # --- Part 3: 3D Body Pose Tracking (Thick Dark Styled) ---
    pose_results = pose_tracker.process(img_rgb)
    body_coords = "No Human Body Detected"
    if pose_results.pose_landmarks:
        # Draw body skeleton with Custom Dark bold specification
        mp_drawing.draw_landmarks(
            image=img_cv, landmark_list=pose_results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=dark_pose_joints, connection_drawing_spec=dark_pose_lines
        )
        lm = pose_results.pose_landmarks.landmark
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        body_coords = (
            f"👕 Left Shoulder: X={(l_sh.x-0.5)*200:.1f}, Y={(l_sh.y-0.5)*200:.1f}, Z={l_sh.z*100:.1f}\n"
            f"👕 Right Shoulder: X={(r_sh.x-0.5)*200:.1f}, Y={(r_sh.y-0.5)*200:.1f}, Z={r_sh.z*100:.1f}\n"
            f"🖐️ Left Wrist: X={(l_wr.x-0.5)*200:.1f}, Y={(l_wr.y-0.5)*200:.1f}, Z={l_wr.z*100:.1f}\n"
            f"🖐️ Right Wrist: X={(r_wr.x-0.5)*200:.1f}, Y={(r_wr.y-0.5)*200:.1f}, Z={r_wr.z*100:.1f}"
        )

    # --- Part 4: Object Detection (Thick Dark Green Boxes) ---
    obj_results = obj_detector(image)
    object_coords = "No Standard Objects Detected"
    if obj_results:
        object_coords = ""
        for obj in obj_results:
            if obj['score'] > 0.5:
                box = obj['box']
                label = obj['label']
                xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                
                # Draw thick Bounding Box (Color: Dark Green (0, 100, 0), Thickness: 3)
                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 100, 0), 3)
                
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                object_coords += f"📦 {label.capitalize()}: Center X={int(cx)}, Center Y={int(cy)} (Conf: {obj['score']*100:.0f}%)\n"
                
                # Dark Text Shadow representation
                cv2.putText(img_cv, f"{label.upper()}", (xmin, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2, cv2.LINE_AA)
        if not object_coords: object_coords = "No high-confidence objects found."

    # Render Output
    final_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(final_img)

    return ai_answer, face_coords, body_coords, object_coords, final_pil

# --- Gradio Clean Dashboard Layout ---
with gr.Blocks() as demo:
    gr.Markdown("Ultimate AI Grounding")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Any Image")
            q_input = gr.Textbox(label=" Ask AI anything about the Image", value="What items are present here?")
            t_input = gr.Textbox(label="Face Parts to Track", value="left eye, right eye, nose, mouth")
            btn = gr.Button("Run Multi-Engine Detection", variant="primary")
            
        with gr.Column(scale=1):
            out_chat = gr.Textbox(label="AI Visual Chat Answer", lines=2)
            
            with gr.Tab("3D Face Coordinates"):
                out_face = gr.TextArea(label="Face Space Positions", lines=4)
            with gr.Tab(" 3D Body Pose Coordinates"):
                out_body = gr.TextArea(label="Main Skeleton Joints", lines=4)
            with gr.Tab(" General Object Tracking"):
                out_objects = gr.TextArea(label="Detected Items Centers (X, Y)", lines=4)
                
            out_img = gr.Image(label="Combined Detection Canvas (Dark Bold Mesh & Boxes)")

    btn.click(
        process_ultimate_grounding, 
        inputs=[img_input, q_input, t_input], 
        outputs=[out_chat, out_face, out_body, out_objects, out_img]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
