'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
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
# Author List:      Priyansh Miyani 
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import math
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        A single BGR frame read from the video, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "center_x" : int,   x-pixel of the lane centre in this frame,
                                or -1 if the lane could not be found
            "lane"     : str,   "left", "right" or "unknown"
        }

    Example call:
    ---
    result = detect_lane(frame)

    COORDINATE SYSTEM:
    ---
    `center_x` is an absolute pixel column in the frame AS RECEIVED - the
    dataset's own resolution, 640x480. It is compared against a ground truth
    measured in those pixels, so it only means anything in them.

    You may resize, crop or warp all you like inside this function, but scale
    the answer back before returning it. A centre found in a 320x240 copy is
    half the value it should be, and a centre read off a bird's-eye view is in
    warped coordinates, not frame ones - map the point back through the inverse
    of your transform. Do not re-encode or resize the clip files themselves.

    NOTE:
    ---
    This function must ONLY compute and return the result.
    Do not call cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from
    inside it. All visualisation and debugging output belongs outside this
    function - see draw_overlay() and process_video() below.
    '''

    center_x = -1
    lane = LANE_UNKNOWN

    #################### ADD YOUR CODE HERE ####################
    h, w = frame.shape[:2]

    # ============================================================
    # TEMPORAL STATE
    # ============================================================
    if not hasattr(detect_lane, "_prev_center"):
        detect_lane._prev_center = None
        detect_lane._prev_lane = None
        detect_lane._velocity = 0.0
        detect_lane._confidence = 0.0

    prev_center = detect_lane._prev_center
    prev_lane = detect_lane._prev_lane
    prev_velocity = detect_lane._velocity

    # ============================================================
    # 1. COLOUR SEGMENTATION
    # ============================================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(
        hsv,
        np.array([10, 55, 55]),
        np.array([52, 255, 255])
    )

    white = cv2.inRange(
        hsv,
        np.array([0, 0, 135]),
        np.array([180, 115, 255])
    )

    top = int(h * 0.36)

    yellow[:top, :] = 0
    white[:top, :] = 0

    white[int(h * 0.88):, :] = 0
    yellow[int(h * 0.96):, :] = 0

    kernel = np.ones((3, 3), np.uint8)

    yellow = cv2.morphologyEx(
        yellow,
        cv2.MORPH_OPEN,
        kernel
    )
    yellow = cv2.morphologyEx(
        yellow,
        cv2.MORPH_CLOSE,
        kernel
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        kernel
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ============================================================
    # 2. ROW GROUP DETECTOR
    # ============================================================
    def row_groups(mask, yy, max_width=100):

        y0 = max(0, yy - 4)
        y1 = min(h, yy + 5)

        count = np.count_nonzero(
            mask[y0:y1, :],
            axis=0
        )

        xs = np.where(count >= 2)[0]

        if len(xs) == 0:
            return []

        groups = []

        start = int(xs[0])
        previous = int(xs[0])

        for value in xs[1:]:

            value = int(value)

            if value - previous > 7:

                width_group = previous - start + 1

                if 1 <= width_group <= max_width:
                    groups.append(
                        (start + previous) * 0.5
                    )

                start = value

            previous = value

        width_group = previous - start + 1

        if 1 <= width_group <= max_width:
            groups.append(
                (start + previous) * 0.5
            )

        return groups

    # ============================================================
    # 3. DENSE SCANLINES
    # ============================================================
    track_rows = [
        int(h * r)
        for r in (
            0.40, 0.43, 0.46, 0.49,
            0.52, 0.55, 0.58, 0.61,
            0.64, 0.67, 0.70, 0.73,
            0.76, 0.79, 0.82, 0.85
        )
    ]

    # ============================================================
    # 4. WHITE DIVIDER
    # ============================================================
    white_points = []

    for yy in track_rows:

        candidates = row_groups(
            white,
            yy,
            85
        )

        candidates = [
            x for x in candidates
            if 0.03 * w < x < 0.97 * w
        ]

        if not candidates:
            continue

        if prev_center is not None:

            expected = prev_center

            def score_white(x):
                return (
                    0.40 * abs(x - w * 0.50)
                    +
                    0.60 * abs(x - expected)
                )

            chosen = min(
                candidates,
                key=score_white
            )

        else:

            chosen = min(
                candidates,
                key=lambda x: abs(x - w * 0.50)
            )

        white_points.append(
            (yy, float(chosen))
        )

    # ============================================================
    # 5. ROBUST WHITE CURVE FIT
    # ============================================================
    white_coeff = None

    if len(white_points) >= 3:

        ys = np.asarray(
            [p[0] for p in white_points],
            dtype=np.float32
        )

        xs = np.asarray(
            [p[1] for p in white_points],
            dtype=np.float32
        )

        try:

            if len(white_points) >= 5:

                c = np.polyfit(
                    ys,
                    xs,
                    2
                )

                pred = np.polyval(
                    c,
                    ys
                )

                residual = np.abs(
                    xs - pred
                )

                med = float(
                    np.median(residual)
                )

                limit = max(
                    22.0,
                    med * 2.8
                )

                keep = residual <= limit

                if np.count_nonzero(keep) >= 4:

                    white_coeff = np.polyfit(
                        ys[keep],
                        xs[keep],
                        2
                    )

                else:
                    white_coeff = c

            else:

                white_coeff = np.polyfit(
                    ys,
                    xs,
                    1
                )

        except Exception:
            white_coeff = None

    # ============================================================
    # 6. LANE IDENTIFICATION
    # ============================================================
    if white_coeff is not None:

        divider_positions = []

        for r in (
            0.70,
            0.74,
            0.78,
            0.82
        ):

            x = float(
                np.polyval(
                    white_coeff,
                    int(h * r)
                )
            )

            if 0 <= x < w:
                divider_positions.append(x)

        if divider_positions:

            divider_x = float(
                np.median(divider_positions)
            )

            if divider_x >= w * 0.50:
                lane = LANE_LEFT
            else:
                lane = LANE_RIGHT

    # ============================================================
    # 7. YELLOW BOUNDARY
    # ============================================================
    yellow_points = []

    if white_coeff is not None:

        last_yellow = None

        for yy in track_rows:

            candidates = row_groups(
                yellow,
                yy,
                125
            )

            if not candidates:
                continue

            white_x = float(
                np.polyval(
                    white_coeff,
                    yy
                )
            )

            if lane == LANE_LEFT:

                valid = [
                    x for x in candidates
                    if x < white_x - 18
                ]

            elif lane == LANE_RIGHT:

                valid = [
                    x for x in candidates
                    if x > white_x + 18
                ]

            else:
                continue

            if not valid:
                continue

            if last_yellow is None:

                chosen = min(
                    valid,
                    key=lambda x:
                    abs(abs(white_x - x) - 180)
                )

            else:

                chosen = min(
                    valid,
                    key=lambda x:
                    abs(x - last_yellow)
                )

                if abs(chosen - last_yellow) > 130:

                    alternative = min(
                        valid,
                        key=lambda x:
                        abs(abs(white_x - x) - 180)
                    )

                    if abs(
                        alternative - last_yellow
                    ) <= 160:
                        chosen = alternative
                    else:
                        continue

            yellow_points.append(
                (yy, float(chosen))
            )

            last_yellow = float(chosen)

    # ============================================================
    # 8. FIT YELLOW CURVE
    # ============================================================
    yellow_coeff = None

    if len(yellow_points) >= 3:

        ys = np.asarray(
            [p[0] for p in yellow_points],
            dtype=np.float32
        )

        xs = np.asarray(
            [p[1] for p in yellow_points],
            dtype=np.float32
        )

        try:

            if len(yellow_points) >= 5:

                c = np.polyfit(
                    ys,
                    xs,
                    2
                )

                pred = np.polyval(
                    c,
                    ys
                )

                residual = np.abs(
                    xs - pred
                )

                med = float(
                    np.median(residual)
                )

                limit = max(
                    25.0,
                    med * 2.8
                )

                keep = residual <= limit

                if np.count_nonzero(keep) >= 4:

                    yellow_coeff = np.polyfit(
                        ys[keep],
                        xs[keep],
                        2
                    )

                else:
                    yellow_coeff = c

            else:

                yellow_coeff = np.polyfit(
                    ys,
                    xs,
                    1
                )

        except Exception:
            yellow_coeff = None

    # ============================================================
    # 9. V19 ADAPTIVE GEOMETRIC CENTRE
    #
    # White divider remains the primary geometry.
    # Yellow is used to estimate lane width, but unreliable
    # heights are rejected before calculating the centre.
    # ============================================================
    centre_candidates = []

    if (
        white_coeff is not None
        and yellow_coeff is not None
        and lane in (
            LANE_LEFT,
            LANE_RIGHT
        )
    ):

        evaluation = [
            (0.60, 0.05),
            (0.64, 0.07),
            (0.68, 0.10),
            (0.72, 0.14),
            (0.76, 0.18),
            (0.80, 0.21),
            (0.83, 0.15),
            (0.85, 0.07),
            (0.87, 0.03)
        ]

        widths = []

        samples = []

        for ratio, base_weight in evaluation:

            yy = int(h * ratio)

            wx = float(
                np.polyval(
                    white_coeff,
                    yy
                )
            )

            yx = float(
                np.polyval(
                    yellow_coeff,
                    yy
                )
            )

            width_lane = abs(wx - yx)

            if not (
                75 <= width_lane <= 400
            ):
                continue

            if lane == LANE_LEFT:

                if yx >= wx:
                    continue

            else:

                if yx <= wx:
                    continue

            centre = (
                wx + yx
            ) * 0.5

            if not (
                20 <= centre <= w - 20
            ):
                continue

            widths.append(width_lane)

            samples.append(
                (
                    ratio,
                    base_weight,
                    wx,
                    yx,
                    width_lane,
                    centre
                )
            )

        if samples:

            # Robust lane-width estimate.
            width_array = np.asarray(
                widths,
                dtype=np.float32
            )

            width_median = float(
                np.median(width_array)
            )

            width_deviation = float(
                np.median(
                    np.abs(
                        width_array -
                        width_median
                    )
                )
            )

            if width_deviation < 1.0:
                width_deviation = 1.0

            for (
                ratio,
                base_weight,
                wx,
                yx,
                width_lane,
                centre
            ) in samples:

                # Width consistency.
                width_error = abs(
                    width_lane -
                    width_median
                )

                consistency = 1.0 / (
                    1.0 +
                    width_error /
                    max(
                        12.0,
                        width_deviation * 2.5
                    )
                )

                # Perspective prior:
                # avoid trusting extremely wide/narrow
                # detections at the extreme rows.
                expected_width = (
                    125.0 +
                    125.0 *
                    ((ratio - 0.60) / 0.27)
                )

                perspective_score = 1.0 - min(
                    1.0,
                    abs(
                        width_lane -
                        expected_width
                    ) / 180.0
                )

                # Temporal centre agreement.
                temporal_score = 1.0

                if (
                    prev_center is not None
                    and prev_lane == lane
                ):

                    temporal_score = 1.0 - min(
                        1.0,
                        abs(
                            centre -
                            float(prev_center)
                        ) / 180.0
                    )

                    temporal_score = (
                        0.55 +
                        0.45 * temporal_score
                    )

                weight = (
                    base_weight
                    *
                    (
                        0.45
                        +
                        0.35 * consistency
                        +
                        0.20 * perspective_score
                    )
                    *
                    temporal_score
                )

                centre_candidates.append(
                    (
                        centre,
                        weight,
                        width_lane,
                        ratio
                    )
                )

    # ============================================================
    # 10. ROBUST CENTRE ESTIMATION
    # ============================================================
    if centre_candidates:

        centres = np.asarray(
            [p[0] for p in centre_candidates],
            dtype=np.float32
        )

        weights = np.asarray(
            [p[1] for p in centre_candidates],
            dtype=np.float32
        )

        # First robust estimate.
        order = np.argsort(centres)

        centres_sorted = centres[order]
        weights_sorted = weights[order]

        cumulative = np.cumsum(
            weights_sorted
        )

        total_weight = float(
            cumulative[-1]
        )

        idx = int(
            np.searchsorted(
                cumulative,
                total_weight * 0.5
            )
        )

        idx = min(
            idx,
            len(centres_sorted) - 1
        )

        robust_center = float(
            centres_sorted[idx]
        )

        # Reject centres which disagree strongly
        # with the main geometric solution.
        good = [
            p for p in centre_candidates
            if abs(
                p[0] -
                robust_center
            ) <= 35
        ]

        if len(good) >= 2:

            numerator = sum(
                p[0] * p[1]
                for p in good
            )

            denominator = sum(
                p[1]
                for p in good
            )

            if denominator > 0:

                center_x = int(
                    round(
                        numerator /
                        denominator
                    )
                )

        elif centre_candidates:

            center_x = int(
                round(
                    robust_center
                )
            )

    # ============================================================
    # 11. WHITE + ROBUST MEASURED WIDTH FALLBACK
    # ============================================================
    if (
        center_x < 0
        and white_coeff is not None
        and lane in (
            LANE_LEFT,
            LANE_RIGHT
        )
    ):

        measured_widths = []

        if yellow_coeff is not None:

            for ratio in (
                0.68,
                0.72,
                0.76,
                0.80,
                0.84
            ):

                yy = int(
                    h * ratio
                )

                wx = float(
                    np.polyval(
                        white_coeff,
                        yy
                    )
                )

                yx = float(
                    np.polyval(
                        yellow_coeff,
                        yy
                    )
                )

                width_lane = abs(
                    wx - yx
                )

                if 75 <= width_lane <= 360:

                    if lane == LANE_LEFT:
                        valid_side = yx < wx
                    else:
                        valid_side = yx > wx

                    if valid_side:
                        measured_widths.append(
                            width_lane
                        )

        if measured_widths:

            lane_width = float(
                np.median(
                    measured_widths
                )
            )

        else:
            lane_width = 190.0

        lane_width = float(
            np.clip(
                lane_width,
                85.0,
                250.0
            )
        )

        target_y = int(
            h * 0.80
        )

        wx = float(
            np.polyval(
                white_coeff,
                target_y
            )
        )

        if lane == LANE_LEFT:

            center_x = int(
                round(
                    wx -
                    lane_width * 0.5
                )
            )

        else:

            center_x = int(
                round(
                    wx +
                    lane_width * 0.5
                )
            )

    # ============================================================
    # 12. V19 TEMPORAL FILTER
    # ============================================================
    if (
        center_x >= 0
        and lane in (
            LANE_LEFT,
            LANE_RIGHT
        )
    ):

        current = float(center_x)

        if (
            prev_lane is not None
            and prev_lane != lane
        ):

            filtered = current
            velocity = 0.0

        elif prev_center is None:

            filtered = current
            velocity = 0.0

        else:

            difference = (
                current -
                float(prev_center)
            )

            # Predicted position.
            predicted = (
                float(prev_center)
                +
                prev_velocity
            )

            prediction_error = abs(
                current -
                predicted
            )

            if prediction_error <= 12:

                # Very stable geometry.
                filtered = (
                    0.25 * current
                    +
                    0.75 * predicted
                )

            elif prediction_error <= 28:

                filtered = (
                    0.45 * current
                    +
                    0.55 * predicted
                )

            elif abs(difference) <= 55:

                # Normal road movement / curve.
                filtered = (
                    0.75 * current
                    +
                    0.25 * float(prev_center)
                )

            elif abs(difference) <= 90:

                # Real turn: allow geometry to dominate.
                filtered = (
                    0.90 * current
                    +
                    0.10 * float(prev_center)
                )

            else:

                # Very large movement.
                # Trust current geometry rather than
                # dragging an old centre into the new frame.
                filtered = current

            velocity = (
                0.65 * prev_velocity
                +
                0.35 * difference
            )

        filtered = float(
            np.clip(
                filtered,
                20,
                w - 20
            )
        )

        center_x = int(
            round(filtered)
        )

        detect_lane._prev_center = filtered
        detect_lane._prev_lane = lane
        detect_lane._velocity = velocity

    # ============================================================
    # 13. FINAL VALIDATION
    # ============================================================
    if lane not in (
        LANE_LEFT,
        LANE_RIGHT
    ):

        lane = LANE_UNKNOWN
        center_x = -1

        detect_lane._prev_center = None
        detect_lane._prev_lane = None
        detect_lane._velocity = 0.0
        detect_lane._confidence = 0.0

    elif (
        center_x < 20
        or center_x >= w - 20
    ):

        center_x = -1

    return {
        "center_x": int(center_x),
        "lane": lane
    }
    
    ############################################################

    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
