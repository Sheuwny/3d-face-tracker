import gradio as gr
import torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection, BlipProcessor, BlipForConditionalGeneration

# Force CPU
device = "cpu"

print("Loading models on CPU... (Yeh thoda time lega)")
# BLIP-1 load karna (Free CPU ke liye perfect aur light hai, ~1GB)
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", low_cpu_mem_usage=True).to(device)

# OWL-ViT load karna (With low memory usage)
owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", low_cpu_mem_usage=True).to(device)

def grounded_chat(image, question, target_object):
    if image is None: return "Image upload karo!", None
    
    # 1. Chat
    prompt = f"Question: {question} Answer:"
    inputs = blip_proc(images=image, text=prompt, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=50)
    answer = blip_proc.decode(out[0], skip_special_tokens=True)
    
    # 2. Grounding
    inputs = owl_proc(text=[[target_object]], images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = owl_model(**inputs)
    
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = owl_proc.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    boxes = results[0]["boxes"]
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=5)
        
    return answer, img_copy

# Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Visual Chat (CPU Mode)")
    img_input = gr.Image(type="pil")
    q_input = gr.Textbox(label="Question")
    t_input = gr.Textbox(label="Object to highlight")
    btn = gr.Button("Analyze")
    out_text = gr.Textbox(label="AI Answer")
    out_img = gr.Image(label="Grounding Result")
    
    btn.click(grounded_chat, inputs=[img_input, q_input, t_input], outputs=[out_text, out_img])

demo.launch()
