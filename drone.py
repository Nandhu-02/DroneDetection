import cv2
from ultralytics import YOLO
import numpy as np
import time

def process_image(image_path, model):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Run YOLOv8 inference on the image
    results = model(image)
    
    # Visualize the results on the image
    annotated_frame = results[0].plot()
    
    return annotated_frame

def process_video(video_path, model, output_path=None):
    # Open the video file or webcam
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process video frames
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("Drone Detection", annotated_frame)
        
        # Write the frame to output video if specified
        if output_path:
            out.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')  # or use your custom trained model
    
    # Process image
    image_path = r"C:\Users\Pravallika\OneDrive\Desktop\drone1.png"
    result_image = process_image(image_path, model)
    if result_image is not None:
        cv2.imshow('Drone Detection - Image', result_image)
        cv2.waitKey(0)
        cv2.imwrite('output_image.jpg', result_image)
    


if __name__ == '__main__':
    main()