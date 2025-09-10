import os
import json
import base64
import tempfile
import boto3
import torch
import numpy as np
from PIL import Image
import uuid

ASU_ID = "1233433957"
MODEL_PATH = "/var/task/resnetV1.pt"
MODEL_WT_PATH = "/var/task/resnetV1_video_weights.pt"

sqs = boto3.client("sqs", region_name="us-east-1")
response_queue_url = sqs.get_queue_url(QueueName=f"{ASU_ID}-resp-queue")["QueueUrl"]

class face_recognition:
    def __init__(self, model_path, model_wt_path):
        self.resnet = torch.jit.load(model_path).eval()
        saved_data = torch.load(model_wt_path, map_location="cpu")
        self.embedding_list = saved_data[0]
        self.name_list = saved_data[1]

    def face_recognition_func(self, face_img_path):

        # Step 1: Load image as PIL
        face_pil = Image.open(face_img_path).convert("RGB")
        key      = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]

        # Step 2: Convert PIL to NumPy array (H, W, C) in range [0, 255]
        face_numpy = np.array(face_pil, dtype=np.float32)  # Convert to float for scaling

        # Step 3: Normalize values to [0,1] and transpose to (C, H, W)
        face_numpy /= 255.0  # Normalize to range [0,1]

        # Convert (H, W, C) → (C, H, W)
        face_numpy = np.transpose(face_numpy, (2, 0, 1))

        # Step 4: Convert NumPy to PyTorch tensor
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        if face_tensor != None:
            emb             = self.resnet(face_tensor.unsqueeze(0)).detach()  # detech is to make required gradient false
            dist_list       = []  # list of matched distances, minimum distance is used to identify the person

            for idx, emb_db in enumerate(self.embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
            return self.name_list[idx_min]
        else:
            print(f"No face is detected")
            return

recognizer = face_recognition(MODEL_PATH, MODEL_WT_PATH)

def handler(event, context):
    try:
        for record in event["Records"]:
            body = json.loads(record["body"])
            request_id = body.get("request_id")
            face_b64 = body.get("face_image")

            if not (request_id and face_b64):
                continue

            tmp_path = f"/tmp/{uuid.uuid4().hex}.jpg"
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(face_b64))

            # recognizer = face_recognition(MODEL_PATH, MODEL_WT_PATH)
            prediction = recognizer.face_recognition_func(tmp_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            if prediction is None:
                continue

            response = {
                "request_id": request_id,
                "result": prediction
            }

            sqs.send_message(
                QueueUrl=response_queue_url,
                MessageBody=json.dumps(response)
            )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Recognition done."})
        }

    except Exception as e:
        print("Recognition failed:", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
