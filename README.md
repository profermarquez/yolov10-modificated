#  Se modifico:

Se cambio la version de scipy y se le puso sin version

Probado en un entorno virtual con Python 3.8.7

Funciona con la libreria gradio.



## Instalacion
/virtualenv env       /env/Scripts/activate.bat
`conda` virtual environment is recommended. 
```
conda create -n yolov10 python=3.9
conda activate yolov10
pip install -r requirements.txt
pip install -e .
```
## Demo
```
python app.py
# Luego de ejecutar visitar http://127.0.0.1:7860
```
## Entrenamiento

from ultralytics import YOLOv10

model = YOLOv10()

model.train(data='coco.yaml', epochs=500, batch=256, imgsz=640)
```

## Citacion

If our code or models help your work, please cite our paper:
```BibTeX
@article{wang2024yolov10,
  title={YOLOv10: Real-Time End-to-End Object Detection},
  author={Wang, Ao and Chen, Hui and Liu, Lihao and Chen, Kai and Lin, Zijia and Han, Jungong and Ding, Guiguang},
  journal={arXiv preprint arXiv:2405.14458},
  year={2024}
}
```
