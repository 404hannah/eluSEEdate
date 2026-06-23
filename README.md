# eluSEEdate

**Assistive Navigation System with Obstacle Detection using ConvLSTM and YOLOv12**

### Summary

EluSEEdate is a mobile application for visually-impaired people that provides the obstacle and the direction (front, left, right) the user should take for both destination-based and destination-free navigation. However, a significant increase in the accuracy of the models is necessary before real-world deployment.

### Features

1. Real-Time Obstacle Detection
2. Destination-Based Navigation
3. Destination-Free Navigation
4. Directional audio feedback and guidance
5. Voice Activated System Navigation

### Results

**Table 1. Overall ConvLSTM Scores**
| Metrics | ConvLSTM |
| --------- | ------------- |
| Accuracy | 51.25% |
| Precision | 53.38% |
| Recall | 51.25% |
| Latency | 37.90 ms/clip |

**Table 2. ConvLSTM Modes**
| Mode | Accuracy |
| ----------------- | -------- |
| Both | 51.25% |
| Destination-based | 56.88% |
| Destination-free | 45.45% |

**Table 3. ConvLSTM Directions**
| Directions | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Front | 0.66 | 0.52 | 0.58 |
| Left | 0.42 | 0.56 | 0.48 |
| Right | 0.46 | 0.46 | 0.46 |

**Table 4. YOLO Scores**
| Metrics | YOLOv12N |
| ----------------- | -------- |
| mAP IoU 0.50 | 0.559 |
| mAP IoU 0.50:0.95 | 0.403 |
| AR IoU 0.50:0.95 | 0.604 |

### Main References

- A. Jadhav, J. Cao, A. Shetty, U. Kumar, A. Sharma, B. Sukboontip, J. Tamarapalli, J.
  Zhang, and A. Koul, “AI Guide Dog: Egocentric path prediction on smartphone,”
  Proceedings of the AAAI Symposium Series, 2025, vol. 5, no. 1 pp. 220-227,
  https://doi.org/10.1609/aaaiss.v5i1.35591.

- Y. Tian, Q. Ye, and D. Doermann, “YOLOV12: Attention-Centric Real-Time Object
  Detectors,” arXiv.org, https://doi.org/10.48550/arXiv.2502.12524.
