# A simple falling detection system based on mediapipe

## 1. Training
```sh
python train.py
```

## 2. Predict Image
```sh
# change link to img in predict_img.py, then run command
python predict_img.py
```

## 3. Predict Webcam/ Video
### 3.1. Predict with Webcam
```sh
python predict_video.py
```

### 3.2. Predict Video
```
# change input source of cv2 to link video, then run command
python predict.py
```

## 4. Demo
![image](https://user-images.githubusercontent.com/19906050/223056459-75083904-f78e-48b8-923d-76ff3c449b93.png)

## 5. Deploy to Web
To be able to use webcam for web, you must install ssl for your web. Then run below command:
```sh
cd deploy_web
python app.py
```
