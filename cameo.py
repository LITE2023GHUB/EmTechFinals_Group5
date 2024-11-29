import cv2
import numpy as np
import filters  # Assuming filters contains the necessary processing functions
from managers import WindowManager, CaptureManager
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class Cameo:
    def __init__(self, input_path):
        self._input_path = input_path
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = None
        self.kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])

    def run(self):
        # Determine input type: image or video
        if self._input_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            self.process_image()
        elif self._input_path.lower().endswith(('avi', 'mp4', 'mkv', 'mov')):
            self.process_video()
        else:
            print("Unsupported file format")

    def process_image(self):
        frame = cv2.imread(self._input_path)
        if frame is None:
            print("Failed to load image.")
            return
        
        self._windowManager.createWindow()
        filters.detect_face(frame)  # Apply the filter or detection logic
        cv2.imshow('Cameo', frame)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()

    def process_video(self):
        self._captureManager = CaptureManager(cv2.VideoCapture(self._input_path), self._windowManager, False)
        self._windowManager.createWindow()
        
        while self._captureManager._capture.isOpened():
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is None:
                break
            filters.detect_face(frame)  # Apply the filter or detection logic
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
        
        self._captureManager._capture.release()
        cv2.destroyAllWindows()

    def onKeypress(self, keycode):
        if keycode == 32:  # Space key
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # Tab key
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # Escape key
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    # Create a GUI file picker
    root = Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user to select a file
    file_path = askopenfilename(title="Select Image or Video File",
                                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                                           ("Video files", "*.avi *.mp4 *.mkv *.mov")])

    if file_path:
        Cameo(file_path).run()
    else:
        print("No file selected.")
