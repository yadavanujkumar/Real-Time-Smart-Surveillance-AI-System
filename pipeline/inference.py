import cv2
import time
from core.video_input import VideoInput
from core.detection import ObjectDetectorTracker
from core.face_recognition import FaceRecognizer
from core.behavior_analysis import BehaviorAnalyzer
from core.alert_engine import AlertEngine
from utils.logger import logger

class InferencePipeline:
    """Orchestrates the entire AI pipeline: Video -> Detection -> Tracking -> Analysis -> Output"""
    
    def __init__(self, source=0):
        self.video_input = VideoInput(source)
        self.detector = ObjectDetectorTracker()
        self.face_recognizer = FaceRecognizer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.alert_engine = AlertEngine()
        self.is_running = False

    def run(self):
        """Starts the real-time inference loop."""
        self.video_input.start()
        self.is_running = True
        logger.info("Starting inference pipeline loop.")

        prev_time = time.time()

        try:
            while self.is_running:
                ret, frame = self.video_input.read_frame()
                if not ret:
                    logger.warning("No frame read from video source. Exiting loop.")
                    break

                # 1. Detection & Tracking
                annotated_frame, detections = self.detector.process_frame(frame)

                # 2. Behavior Analysis
                behavior_events = self.behavior_analyzer.process(annotated_frame, detections)
                for event in behavior_events:
                    self.alert_engine.log_event(
                        event_type=event["type"],
                        track_id=event["track_id"],
                        details=f"BBox: {event['bbox']}"
                    )

                # 3. Face Recognition
                recognized_faces = self.face_recognizer.process(annotated_frame, detections)

                # Output FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Show the frame
                cv2.imshow("Smart Surveillance AI System", annotated_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit signal received.")
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        except Exception as e:
            logger.error(f"Error in inference pipeline: {e}")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        self.video_input.stop()
        cv2.destroyAllWindows()
        logger.info("Inference pipeline stopped.")
