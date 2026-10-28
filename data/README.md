# Data folder

Not committed to Git (see `.gitignore`). To reproduce:

1. Download the YOLOv8 export of *Pork Rasher Error(Packaging) v4*
   (https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4, CC BY 4.0),
   or use the ZIP supplied on Moodle: `Pork_Rasher_Error-Packaging-_v4i_yolov8.zip`.
2. Unzip it so this exists: `data/raw/data.yaml`, `data/raw/train/images`, ...
3. `python scripts/02_make_grouped_split.py` creates `data/grouped/`.
