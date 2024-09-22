import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import torch  # Asegúrate de importar torch para verificar si CUDA está disponible

# Verifica si CUDA está disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_edge_detection(image):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar el filtro de bordes Canny
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    # Convertir la imagen de bordes a formato BGR para mantener la consistencia
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    # Cargar el modelo y moverlo a la GPU si está disponible
    model = YOLOv10.from_pretrained(f'jameslahm/{model_id}').to(device)
    
    if image:
        # Procesar imagen con GPU
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, device=device)
        annotated_image = results[0].plot()

        # Aplicar el filtro de bordes a la imagen anotada
        edges_annotated_image = apply_edge_detection(annotated_image)
        
        return edges_annotated_image[:, :, ::-1], None
    else:
        # Procesar video con GPU
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mover el frame a la GPU antes de predecir
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, device=device)
            annotated_frame = results[0].plot()

            # Aplicar el filtro de bordes a cada frame anotado
            edges_annotated_frame = apply_edge_detection(annotated_frame)
            
            out.write(edges_annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path

# Mantén el resto del código igual para que funcione con la inferencia GPU
def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image with Edges", visible=True)
                output_video = gr.Video(label="Annotated Video with Edges", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible(False))
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible(False))
            output_video = gr.update(visible(False)) if input_type == "Image" else gr.update(visible(True))

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold)

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection with Edge Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        YOLOv10 with Canny Edge Detection
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()
