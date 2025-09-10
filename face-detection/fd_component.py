import os
import json
import boto3
import base64
import io
import numpy as np
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), 'facenet_pytorch'))
from facenet_pytorch import MTCNN

# === CONFIG ===
ASU_ID = "1233433957"
TOPIC = f"clients/{ASU_ID}-IoTThing"
SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/810720486914/1233433957-req-queue"

# === AWS Clients ===
sqs_client = boto3.client("sqs", region_name="us-east-1")

# === Face Detector ===
face_detector = MTCNN(image_size=240, margin=0, min_face_size=20, keep_all=False)

# === MQTT Listener (Simulated) ===
def lambda_handler(event, context=None):
    try:
        print("Received message for processing.")
        message = json.loads(event)

        # Extract fields
        encoded = message.get("encoded")
        request_id = message.get("request_id")
        filename = message.get("filename")

        if not (encoded and request_id and filename):
            print("Invalid message structure.")
            return

        # Decode base64 image
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Face detection
        face = face_detector(image)

        if face is not None:
            print(f"Face detected in {filename}")
            face_np = face.permute(1, 2, 0).byte().numpy()
            face_pil = Image.fromarray(face_np)

            buffer = io.BytesIO()
            face_pil.save(buffer, format="JPEG")
            face_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            payload = {
                "request_id": request_id,
                "filename": filename,
                "encoded": face_encoded
            }

            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(payload)
            )
            print(f"Sent to SQS: {filename}")
        else:
            print(f"No face detected in {filename}")
            # (Optional) Send "No-Face" to response queue for bonus

    except Exception as e:
        print(f"Error during processing: {str(e)}")

# === Main loop (MQTT listener simulation) ===
if __name__ == "__main__":
    print("fd_component.py is ready. Waiting for MQTT messages.")
    while True:
        pass  # In actual deployment, MQTT messages will call `lambda_handler`
