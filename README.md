
# 🧠 Проект распознавания блюд с помощью YOLOv11

## 📋 Предварительные требования

- Python версии **3.8+**
- Установка зависимостей:
  ```bash
  pip install -r requirements.txt
  ```
- YOLOv11 (через [Ultralytics YOLO](https://github.com/ultralytics/ultralytics))
- Видео с Яндекс.Диска (поместить в `data/raw/`)
- Скрипт `train.py` и конфигурация `cfg/yolov11n.yaml` из репозитория Ultralytics
- Предобученные веса `yolo11n.pt` (скачать через Ultralytics)

---

## 📁 Структура репозитория

```
dish_recognition_project/
├── cfg/
│   ├── yolov11n.yaml
├── data/
│   ├── raw/               # Исходное видео
│   ├── frames/            # Извлечённые кадры
│   ├── annotations/       # Аннотации
│   ├── augmented/         # Аугментированные изображения
│   ├── dataset.yaml       # Конфигурация датасета
├── scripts/
│   ├── extract_frames.py      # Извлечение кадров из видео
│   ├── annotate.py            # Аннотирование изображений
│   ├── augment_data.py        # Аугментация изображений
│   ├── train_yolo.py          # Обучение YOLOv11
│   ├── evaluate.py            # Оценка модели
│   ├── visualize_results.py   # Генерация видео с предсказаниями
├── results/
│   ├── metrics/
│   ├── output_video_exp1.mp4
│   ├── output_video_exp2.mp4
├── train.py
├── yolo11n.pt
├── requirements.txt
├── .gitignore
├── report.md
├── README.md
```

---

## 🚀 Как воспроизвести

1. **Скачать видео** и поместить его в папку `data/raw/`.

2. **Извлечь кадры** из видео:
   ```bash
   python scripts/extract_frames.py --video_path data/raw/restaurant_video.mp4 --output_dir data/frames --fps 1
   ```

3. **Аннотировать кадры вручную**, используя [LabelImg](https://github.com/tzutalin/labelImg) или [CVAT](https://github.com/opencv/cvat). Сохраняем аннотации в `data/annotations/`.

4. **Аугментировать данные**:
   ```bash
   python scripts/augment_data.py
   ```

5. **Обучить модель**:
   Убедитесь, что файлы `train.py`, `cfg/yolov11n.yaml` и `yolo11n.pt` находятся в корне проекта, затем выполните:
   ```bash
   python scripts/train_yolo.py
   ```

6. **Оценить модель**:
   ```bash
   python scripts/evaluate.py
   ```
   Метрики будут сохранены в `results/metrics/evaluation.csv` и `metrics_plot_*.png`.

7. **Визуализировать результаты**:
   ```bash
   python scripts/visualize_results.py
   ```
   Это создаст видеофайлы `output_video_exp1.mp4` и `output_video_exp2.mp4` в папке `results/`.
