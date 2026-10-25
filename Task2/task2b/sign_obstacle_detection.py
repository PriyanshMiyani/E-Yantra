'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          1568
# Author List:      Dev Patel
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: None 


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    '''
    Purpose:
    ---
    Detect every road sign visible in a single image, and report what each
    one is and how large it appears. An image can hold zero, one, or more
    than one sign at once.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `detections` :  [ list of dict ]
        Zero, one, or many entries - one per sign found in the image:
        [
            {
                "label" : str,   one of SIGN_LABELS
                "area"  : int,   that sign's area in pixels
            },
            ...
        ]

    Example call:
    ---
    detections = detect_signs(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    detections = []
    # The evaluator supplies BGR images. HSV makes red and blue regions easier
    # to isolate under the simulator's lighting.
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
        return detections

    image_h, image_w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red borders identify Speed40 circles and School triangles. Two hue ranges
    # are needed because red wraps around the ends of the HSV hue scale.
    red_low = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    count, red_labels, red_stats, _ = cv2.connectedComponentsWithStats(red_mask, 8)
    for component_id in range(1, count):
        x, y, box_w, box_h, pixel_area = map(int, red_stats[component_id, :5])
        if pixel_area < 100 or box_w < 18 or box_h < 18:
            continue

        aspect_ratio = box_w / float(box_h)
        if not (0.55 <= aspect_ratio <= 1.50):
            continue

        fill_ratio = pixel_area / float(box_w * box_h)
        # Obstacles and large solid red graphics are not red-bordered signs.
        if fill_ratio >= 0.45:
            continue

        component_mask = (
            red_labels[y:y + box_h, x:x + box_w] == component_id
        ).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approximation)

        # Triangle-like contours are School signs; round contours are Speed40.
        # The circularity check rejects thin red crosses and other red graphics.
        if vertices <= 4 and 0.35 <= circularity < 0.72:
            label = "School"
        elif circularity >= 0.72 or (vertices >= 5 and circularity >= 0.55):
            label = "Speed40"
        else:
            continue

        detections.append({"label": label, "area": int(round(contour_area))})

    # Parking and Hospital signs have a saturated blue panel and a light frame.
    # Requiring border evidence avoids confusing blue building windows or the
    # blue lettering on the hospital building with a road sign.
    blue_mask = cv2.inRange(hsv, (90, 120, 35), (135, 255, 255))
    white_mask = ((hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 165)).astype(np.uint8)
    count, blue_labels, blue_stats, _ = cv2.connectedComponentsWithStats(blue_mask, 8)
    candidates = []

    for component_id in range(1, count):
        x, y, box_w, box_h, pixel_area = map(int, blue_stats[component_id, :5])
        if pixel_area < 80 or box_w < 14 or box_h < 14:
            continue

        aspect_ratio = box_w / float(box_h)
        if not (0.55 <= aspect_ratio <= 1.55):
            continue

        pad = max(2, int(round(min(box_w, box_h) * 0.10)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image_w, x + box_w + pad)
        y1 = min(image_h, y + box_h + pad)

        # Measure light border pixels outside each blue component's bounding box.
        strips = (
            white_mask[y0:y, x0:x1],
            white_mask[y + box_h:y1, x0:x1],
            white_mask[y:y + box_h, x0:x],
            white_mask[y:y + box_h, x + box_w:x1],
        )
        side_support = [float(np.mean(strip > 0)) if strip.size else 0.0 for strip in strips]
        if sum(side_support) < 0.42 or sum(value >= 0.14 for value in side_support) < 2:
            continue

        component_mask = (
            blue_labels[y:y + box_h, x:x + box_w] == component_id
        ).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        candidates.append({
            "box": (x, y, box_w, box_h),
            "pixels": pixel_area,
            "contour_area": float(cv2.contourArea(contour)),
        })

    # An outline and the blue panel inside it can be separate components. Merge
    # overlapping candidate boxes so that one physical sign yields one result.
    groups = []
    for candidate in candidates:
        x, y, box_w, box_h = candidate["box"]
        merged = False
        for group in groups:
            gx, gy, group_w, group_h = group["box"]
            ix0 = max(x, gx)
            iy0 = max(y, gy)
            ix1 = min(x + box_w, gx + group_w)
            iy1 = min(y + box_h, gy + group_h)
            intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            smaller_box_area = min(box_w * box_h, group_w * group_h)
            if smaller_box_area and intersection / float(smaller_box_area) >= 0.68:
                new_x = min(x, gx)
                new_y = min(y, gy)
                new_right = max(x + box_w, gx + group_w)
                new_bottom = max(y + box_h, gy + group_h)
                group["box"] = (new_x, new_y, new_right - new_x, new_bottom - new_y)
                group["members"].append(candidate)
                merged = True
                break
        if not merged:
            groups.append({"box": candidate["box"], "members": [candidate]})

    for group in groups:
        x, y, box_w, box_h = group["box"]
        # Use the main blue panel for the glyph; use the largest outer contour
        # for area so that white letters inside the sign do not shrink the area.
        main_component = max(group["members"], key=lambda item: item["pixels"])
        mx, my, mw, mh = main_component["box"]
        glyph_crop = white_mask[my:my + mh, mx:mx + mw]
        if glyph_crop.size == 0:
            continue

        glyph = cv2.resize(glyph_crop, (32, 32), interpolation=cv2.INTER_NEAREST) > 0
        if float(np.mean(glyph)) < 0.04:
            continue
        left_lower_stroke = float(np.mean(glyph[20:30, 5:12]))
        right_lower_stroke = float(np.mean(glyph[20:30, 20:27]))

        # H has two vertical strokes continuing through the lower half; P has
        # only its left stem there. A clipped sign at the right edge can show
        # only part of the P, so keep that visible edge case as Parking.
        if x + box_w >= image_w - 2:
            label = "Parking"
        elif right_lower_stroke > max(0.10, left_lower_stroke * 0.45):
            label = "Hospital"
        else:
            label = "Parking"

        area = max(member["contour_area"] for member in group["members"])
        detections.append({"label": label, "area": int(round(area))})

    return detections


def detect_obstacle(image):
    '''
    Purpose:
    ---
    Detect whether the obstacle is visible in a single image, and how large
    it appears.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "present" : bool,   True if the obstacle is visible
            "area"    : int,    its area in pixels, 0 if not present
        }

    Example call:
    ---
    result = detect_obstacle(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    present = False
    area = 0
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
        return {"present": False, "area": 0}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
    red_high = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_low, red_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(red_mask, 8)
    for component_id in range(1, count):
        x, y, box_w, box_h, pixel_area = map(int, stats[component_id, :5])
        if pixel_area < 400 or box_w < 15 or box_h < 15:
            continue

        aspect_ratio = box_w / float(box_h)
        fill_ratio = pixel_area / float(box_w * box_h)

        # The obstacle is a compact, mostly filled red body. Red sign borders,
        # lane-corner markers, and the outlined hospital cross fail this test.
        if 0.55 <= aspect_ratio <= 1.50 and fill_ratio >= 0.66:
            present = True
            area = max(area, pixel_area)

    if not present:
        area = 0
    return {"present": present, "area": area}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
