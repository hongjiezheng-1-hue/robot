"""ArUco marker generation and detection (OpenCV), tolerant of the API change in OpenCV 4.7."""

import cv2


def _dictionary(name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_marker_image(marker_id=0, dictionary="DICT_4X4_50", side_px=400, border_px=100, flip=False):
    """Marker image with a white quiet-zone border, returned as a grayscale uint8 array."""
    d = _dictionary(dictionary)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, marker_id, side_px)
    else:
        img = cv2.aruco.drawMarker(d, marker_id, side_px)
    img = cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=255)
    return img[:, ::-1].copy() if flip else img


def detect_markers(image, dictionary="DICT_4X4_50"):
    """Return (ids, corners); ids is None when nothing is detected."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    d = _dictionary(dictionary)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
    return ids, corners
