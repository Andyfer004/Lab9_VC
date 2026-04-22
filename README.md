# Laboratorio 9 - Task 3

## Descripción

Este proyecto consiste en un prototipo funcional de detección de objetos en tiempo real utilizando **YOLOv8n**, **Ultralytics** y **OpenCV**.  
El sistema procesa un video pregrabado con temática geek, realiza inferencia cuadro por cuadro, dibuja manualmente las cajas delimitadoras y etiquetas de clase, y muestra los FPS en tiempo real.

## Requisitos

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecutar el script con:

```bash
python task3.py
```

## Configuración

En el archivo `task3.py` se utilizaron los siguientes parámetros principales:

| Parámetro | Valor |
|-----------|-------|
| `MODEL_PATH` | `"yolov8n.pt"` |
| `SOURCE` | `"video.mp4"` |
| `CONF` | `0.45` |
| `IOU` | `0.45` |

## Evidencia de funcionamiento

Video de funcionamiento del programa: [Ver video en YouTube](https://www.youtube.com/watch?v=cOfggxx4n24)