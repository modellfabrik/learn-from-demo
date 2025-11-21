import cv2
import os
import numpy as np

# setup folder dir to save images
folder_name = "images"
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name), exist_ok=True)

""" function 1: basic webcam stream """
def stream_webcam():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print("Streaming webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam Stream", frame)
        
        # waitKey (cv2 function): checks every 1ms, if 'q' pressed, break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

""" function 2: capture and save a frame without streaming """
def capture_frame(filename="captured_image.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Captured Frame", frame)
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    else:
        print("Failed to capture frame")
    
    cap.release()
    cv2.destroyAllWindows()

""" function 3: stream in other formats """
def stream_other_formats():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Streaming grayscale + blurred webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        cv2.imshow("Grayscale + Blur Stream", blurred)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

""" function 4: stream and capture """
def stream_and_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'c' to capture image, 'q' to quit.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam Stream", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            count += 1
            filename = os.path.join(images_dir, f"capture_{count}.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        elif key == ord('q'):
            break

""" function 5: stream with shapes and text """
def stream_with_shapes():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Streaming webcam with shapes. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # get shape info of frame:
        height, width = frame.shape[:2]  # shape returns (height, width, channels)
        center_x = width // 2
        center_y = height // 2

        # basic functions to add rectangles, etc. useful for image annotation
        # rectangle
        cv2.rectangle(frame, (200, 200), (300, 300), (0, 255, 0), 3)
        # solid circle at the center
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        # text
        cv2.putText(frame, "hello learn-from-demo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam with Shapes", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

""" function 6: canny edge detection """
def stream_edges():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Streaming webcam with edge detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # dilate edges to merge nearby contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > 500:  # filter smaller bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Edges", edges)
        cv2.imshow("Merged Bounding Boxes", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    """ call/uncomment 1 function at a time """
    stream_webcam()
    #capture_frame()
    #stream_other_formats()
    #stream_and_capture()
    #stream_with_shapes()
    #stream_edges()