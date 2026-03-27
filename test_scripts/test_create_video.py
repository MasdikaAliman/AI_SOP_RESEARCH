import cv2
import mediapipe as mp
import numpy as np
import time
from icecream import ic



def create_stream_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print("Hotkeys:  R = capture reference   Q/ESC = quit")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    recording = False
    out = None

    recording = False
    out = None

    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame grab error")
            break

        frame = cv2.flip(frame, 1)

        if recording and out is not None:
            out.write(frame)
            cv2.putText(frame, "REC", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Visual", frame)

        key = cv2.waitKey(1) & 0xFF
        # START / STOP recording
        if key == ord('r'):
            recording = not recording

            if recording:
                print("[INFO] Recording started")

                # Ambil ukuran frame
                frame_height, frame_width = frame.shape[:2]

                # Buat VideoWriter baru
                out = cv2.VideoWriter(
                    "1_cycle_step.mp4", fourcc, fps , (frame_width, frame_height)
                )

            else:
                print("[INFO] Recording stopped & saved")
                if out is not None:
                    out.release()
                    out = None

        # EXIT
        elif key == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
if __name__ =="__main__":
    create_stream_video()

