import cv2
import face_recognition
import numpy as np

def recognize_faces(image_path, known_faces):
    # Load the group image
    group_image = face_recognition.load_image_file(image_path)
    group_rgb = cv2.cvtColor(group_image, cv2.COLOR_RGB2BGR)

    # Detect faces
    face_locations = face_recognition.face_locations(group_image)
    face_encodings = face_recognition.face_encodings(group_image, face_locations)

    if not face_encodings:
        print("No faces found in the group image.")
        return

    # Compare each face in the group with known encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([encoding for _, encoding in known_faces], face_encoding, tolerance=0.5)
        name = "Unknown"

        # If a match is found, get the name of the best match
        face_distances = face_recognition.face_distance([encoding for _, encoding in known_faces], face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_faces[best_match_index][0]

        # Draw box and label
        cv2.rectangle(group_rgb, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(group_rgb, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(group_rgb, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Show result
    cv2.imshow("Recognized Faces", group_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_known_faces():
    known_faces = []
    
    # Load Akshay image
    akshay_image = face_recognition.load_image_file("D://face_recognition//images//akshay.jpg")
    akshay_encoding = face_recognition.face_encodings(akshay_image)
    if akshay_encoding:
        known_faces.append(("akshay", akshay_encoding[0]))
    else:
        print("No face found in akshay.jpg")

    # Load Modi image
    modi_image = face_recognition.load_image_file("D://face_recognition//images//modi.jpg")
    modi_encoding = face_recognition.face_encodings(modi_image)
    if modi_encoding:
        known_faces.append(("modi", modi_encoding[0]))
    else:
        print("No face found in modi.jpg")

    return known_faces

# Run the recognition
known_faces = load_known_faces()
recognize_faces("D://face_recognition//images//group1.jpg", known_faces)
