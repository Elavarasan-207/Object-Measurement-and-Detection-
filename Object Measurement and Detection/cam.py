import cv2
from object_detector import *
import numpy as np

# Define reference object dimensions in centimeters
ref_width_cm = 21.0
ref_height_cm = 29.7

# Load Object Detector
detector = HomogeneousBgDetector()

# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()

    if ret:
        # Detect reference object (A4 paper)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding rectangle of the A4 sheet
        a4_rect = None
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            if w > 100 and h > 100:  # Filter out small contours
                a4_rect = rect
                break

        if a4_rect is not None:
            # Get Width and Height of the reference object in pixels
            ref_width_px = max(a4_rect[1][0], a4_rect[1][1])
            ref_height_px = min(a4_rect[1][0], a4_rect[1][1])

            # Get pixel-to-centimeter ratio
            pixel_cm_ratio = max(ref_width_px / ref_width_cm, ref_height_px / ref_height_cm)

            # Display rectangle
            box = cv2.boxPoints(a4_rect)
            box = np.int0(box)
            cv2.polylines(img, [box], True, (0, 255, 255), 2)

            for cnt in detector.detect_objects(img):
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Check if object is inside the A4 sheet
                if box[0][0] <= x <= box[2][0] and box[0][1] <= y <= box[2][1]:
                    # Get Width and Height of the Objects by applying the Ratio pixel to cm
                    object_width = w / pixel_cm_ratio
                    object_height = h / pixel_cm_ratio

                    # Display rectangle
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.polylines(img, [box], True, (255, 0, 0), 2)
                    cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                    cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 40)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
