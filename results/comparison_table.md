| method | split | mAP50 | mAP50-95 | tray recall | tray specificity | bal. acc | ms/img | params (M) |
|---|---|---|---|---|---|---|---|---|
| grouped_yolov8n | grouped | 0.508 | 0.254 | 0.948 | 0.429 | 0.688 | 9.641 | 3.139 |
| grouped_yolov8s | grouped | 0.423 | 0.236 | 0.941 | 0.429 | 0.685 | 10.246 | 11.138 |
| original_yolov8n | original | 0.443 | 0.212 | 0.912 | 0.833 | 0.873 | 9.986 | 3.139 |
| HOG+colour+LogReg baseline | original | n/a | n/a | 0.809 | 0.667 | 0.738 | 5.866 | 0.002 |
| HOG+colour+LogReg baseline | grouped | n/a | n/a | 0.830 | 0.514 | 0.672 | 5.384 | 0.002 |
