import cv2
import numpy as np

def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    A helper to place text on an image using descriptors like 'top_center' or 'bottom_left'
    instead of calculating pixels manually every time.
    """
    if not isinstance(org, tuple):
        # Calculate how big the text is so we can center it
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0
        h, w, *_ = image.shape
        place_h, place_w = org.split("_")

        # Vertical placement
        if place_h == "top":
            org_h = label_height + 5
        elif place_h == "bottom":
            org_h = h - 5
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        # Horizontal placement
        if place_w == "left":
            org_w = 5
        elif place_w == "right":
            org_w = w - label_width - 5
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def draw_matches(img1, kp1, img2, kp2, matches):
    # Standard OpenCV wrapper to draw lines between matching points
    matches = sorted(matches, key=lambda x: x.distance)
    vis_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis_img

def show_images(images, window_name='Image', image_title=None):
    # Helper to show one or multiple images
    if len(images.shape) == 2:
        images = [images]

    for i, image in enumerate(images):
        image_c = image.copy()
        # Convert floats (0-1) to integers (0-255) if needed
        if image_c.dtype != np.uint8:
            if image_c.max() < 1.:
                image_c = image_c * 255
            image_c = image_c.astype(np.uint8)

        if len(image.shape) == 2:
            image_c = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)

        title = image_title if image_title else f"{i}"
        image_c = put_text(image_c, "top_center", title)
        cv2.imshow(window_name, image_c)
        cv2.waitKey(0)