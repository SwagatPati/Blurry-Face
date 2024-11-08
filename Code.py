import cv2
import numpy as np

def blur_face(frame, x, y, w, h):
    """Blur the face region of the frame."""
    face = frame[y:y+h, x:x+w]
    face_blurred = cv2.GaussianBlur(face, (99, 99), 30)
    frame[y:y+h, x:x+w] = face_blurred

def main():
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam (0 for default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return
    
    # Create a VideoWriter object to save the output video (optional)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            blur_face(frame, x, y, w, h)
        
        # Display the frame with blurred faces
        cv2.imshow('Webcam', frame)
        
        # Write the frame to the output video (optional)
        out.write(frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the main function
if __name__ == "__main__":
    main()
