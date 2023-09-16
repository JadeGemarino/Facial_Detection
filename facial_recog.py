import cv2
import face_recognition
# Load the cascade
face_cascade = cv2.CascadeClassifier('detection/haarcascade_frontalface_default.xml')
# To capture video from webcam. 
cap = cv2.VideoCapture(0)



def triangle():
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



reference_image_1 = face_recognition.load_image_file('face_verify/jade.jpg')
reference_image_2 = face_recognition.load_image_file('face_verify/jade2.jpg')
reference_image_3 = face_recognition.load_image_file('face_verify/jeffrey.jpg')
reference_image_4 = face_recognition.load_image_file('face_verify/jeffrey.jpg')

reference_encoding_1 = face_recognition.face_encodings(reference_image_1)[0]
reference_encoding_2 = face_recognition.face_encodings(reference_image_2)[0]
reference_encoding_3 = face_recognition.face_encodings(reference_image_3)[0]
reference_encoding_4 = face_recognition.face_encodings(reference_image_4)[0]

    
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')


while True:
  

    ret, img = cap.read()


    width = 500  # You can change this to your desired width
    height = 500  
    img = cv2.resize(img, (width,height))


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




    face_locations = face_recognition.face_locations(gray)
    face_encodings = face_recognition.face_encodings(gray, face_locations)

  
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)



    

    for face_encoding, face_location in zip(face_encodings, face_locations):
        results = face_recognition.compare_faces([reference_encoding_1], face_encoding)
        results2 = face_recognition.compare_faces([reference_encoding_2], face_encoding)
        results3 = face_recognition.compare_faces([reference_encoding_3], face_encoding)
        results4 = face_recognition.compare_faces([reference_encoding_4], face_encoding)

         # Set a threshold for similarity
        threshold = 0.6

    # Draw rectangle around the face
        top, right, bottom, left = face_location
        if results[0] and results[0] > threshold:
            label = "Jade"
            color = (0, 255, 0)  # Green for verified

        elif results2[0] and results2[0] > threshold:  # Corrected results2[0]
            label = "Jade"
            color = (0, 255, 0)

        elif results3[0] and results3[0] > threshold:
            label = "Gray Bradsk"
            color = (0, 255, 0)
        
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for not verified

    triangle()

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()