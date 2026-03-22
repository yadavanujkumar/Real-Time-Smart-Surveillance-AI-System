import cv2
import face_recognition
import numpy as np
import pickle
from utils.logger import logger
from utils.config import DATA_DIR, MODELS_DIR, FACE_RECOGNITION_TOLERANCE


class FaceRecognizer:
    """Handles face detection, embedding extraction, and face matching using face_recognition."""

    def __init__(self, tolerance: float = FACE_RECOGNITION_TOLERANCE):
        self.tolerance = tolerance
        self.known_face_encodings: list = []
        self.known_face_names: list = []
        self.encodings_file = MODELS_DIR / "face_encodings.pkl"
        self.faces_dir = DATA_DIR / "known_faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.load_encodings()

    def load_encodings(self):
        """Loads face encodings from disk or computes them if none exist."""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                logger.info(f"Loaded {len(self.known_face_names)} face encodings from {self.encodings_file}.")
            except Exception as e:
                logger.error(f"Failed to load user definitions: {e}")
        else:
            logger.info("No known face encodings found. Starting fresh.")
            self.encode_known_faces()

    def encode_known_faces(self):
        """Scans the known_faces directory to calculate embeddings."""
        logger.info(f"Scanning {self.faces_dir} for known faces...")
        new_encodings = []
        new_names = []

        # Assuming standard structure: known_faces/PersonName/image.jpg
        for person_dir in self.faces_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                for img_path in person_dir.glob("*.*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        try:
                            image = face_recognition.load_image_file(str(img_path))
                            # Calculate face encoding
                            encodings = face_recognition.face_encodings(image)
                            if len(encodings) > 0:
                                new_encodings.append(encodings[0])
                                new_names.append(person_name)
                                logger.info(f"Encoded face for {person_name} from {img_path.name}")
                            else:
                                logger.warning(f"No faces found in {img_path}")
                        except Exception as e:
                            logger.error(f"Error processing {img_path}: {e}")

        # Save to disk
        self.known_face_encodings = new_encodings
        self.known_face_names = new_names
        
        with open(self.encodings_file, "wb") as f:
            pickle.dump({"encodings": self.known_face_encodings, "names": self.known_face_names}, f)
        logger.info(f"Saved {len(self.known_face_names)} encodings to {self.encodings_file}")

    def process(self, frame, detections):
        """
        Takes the frame and YOLO detections (persons) to recognize faces within those bounds.
        To improve performance, only process areas where persons are detected.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # We can either let face_recognition find faces everywhere in the frame, 
        # or constrain search boxes using YOLO detections to speed things up.
        # Here we'll do global face detection for simplicity, but could optimize heavily.
        
        # 1. Detect all face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # 2. Extract encodings for those faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            
            # Find best match
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            recognized_faces.append({
                "bbox": [left, top, right, bottom],
                "name": name
            })

            # Draw
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return recognized_faces
