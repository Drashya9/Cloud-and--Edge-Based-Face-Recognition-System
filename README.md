# Cloud-and--Edge-Based-Face-Recognition-System

# Face Edge Project

This project implements **Face Detection** and **Face Recognition** pipelines using AWS services.

- **Face Detection**: Runs as an AWS Greengrass component (`fd_component.py`) to detect faces in input images.
- **Face Recognition**: Runs as an AWS Lambda (`fr_lambda.py`) to identify and classify detected faces.

## Features
- Real-time face detection with MTCNN / OpenCV.
- Recognition pipeline using FaceNet + KNN.
- AWS integration: S3, SQS, Lambda, Greengrass.
- Secure credential handling.

## 🛠️ Tech Stack
- **Python** (Face Detection & Recognition code)
- **AWS** (S3, Lambda, Greengrass, SQS, IAM)
- **OpenCV**, **MTCNN**, **FaceNet**

## Project Structure
- **face-detection/**
  - `fd_component.py`
- **face-recognition/**
  - `fr_lambda.py`

##  Setup
Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/face-edge-project.git
   cd face-edge-project
