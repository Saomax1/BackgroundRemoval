# Bundled data

`face_detection_yunet_2023mar.onnx`: YuNet face detector, used by Head only
(`cutout/core/head.py`). From https://github.com/opencv/opencv_zoo
(models/face_detection_yunet), MIT licence. SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
It is small (230 KB), so it ships inside the app rather than as a model pack.
