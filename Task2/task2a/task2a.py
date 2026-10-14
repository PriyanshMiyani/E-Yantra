'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
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
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: last_detection = last_white_fit, last_yellow_fit , last_center_fit, smooth_center , lane_width, base_lane_width, lane_side , lane_locked, missing_center , last_steering , last_speed , white_center_offset, yellow_center_offset , control_count , smooth_heading 


####################### IMPORT MODULES #######################
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

last_detection = None
last_white_fit = None
last_yellow_fit = None
last_center_fit = None
smooth_center = None

lane_width = 0.0
base_lane_width = 0.0
lane_side = 0
lane_locked = False
missing_center = 0

last_steering = 0.0
last_speed = 0.30
white_center_offset = None
yellow_center_offset = None
control_count = 0
smooth_heading = None

def detect_lane(frame):
    '''
    Purpose:
    ---
    Work out, from a single camera frame, how wrong the vehicle currently is -
    and hand back whatever `compute_control()` needs in order to correct it.

    This is your Task 1C detector, extended. Task 1C answered *where the lane
    centre is* and *which lane the vehicle is in*; a controller cannot use
    either of those directly. It needs to know how far the vehicle is from
    where it should be, and which way it is pointing.

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]
        One BGR frame from the vehicle's camera, of shape (height, width, 3),
        the right way up and in the right colour order.

    Returns:
    ---
    Anything you like - a tuple, a dictionary, your own class. Whatever you
    return here is passed straight to compute_control() and nowhere else, so
    the two functions only have to agree with each other.

    Example call:
    ---
    detection = detect_lane(frame)

    REMEMBER:
    ---
    Both lanes are driven during evaluation, and the arena looks mirrored from
    the other one. Do not hard-code which side the boundary is on.

    Do NOT call cv2.imshow() or cv2.imwrite() from inside this function.

    You are marked on how closely you hold the MIDDLE of your lane, not merely
    on staying the correct side of the dashed line.
    '''

    # ADD YOUR CODE HERE
    global last_detection, last_white_fit, last_yellow_fit
    global last_center_fit, smooth_center, lane_width, lane_side
    global lane_locked, missing_center
    global white_center_offset, yellow_center_offset, base_lane_width, smooth_heading

    if frame is None:
        missing_center += 1
        return None

    h, w = frame.shape[:2]

    if h < 20 or w < 20:
        missing_center += 1
        return None

    image_center = w / 2.0
    target_y = h * 0.68
    horizon_y = h * 0.22

    rows = [
        int(h * r)
        for r in np.arange(0.32, 0.88, 0.02)
    ]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(
        hsv,
        np.array([10, 45, 45]),
        np.array([55, 255, 255])
    )

    white = cv2.inRange(
        hsv,
        np.array([0, 0, 125]),
        np.array([180, 125, 255])
    )

    yellow[:int(h * 0.20), :] = 0
    white[:int(h * 0.20), :] = 0

    yellow[int(h * 0.96):, :] = 0
    white[int(h * 0.92):, :] = 0

    kernel = np.ones((3, 3), np.uint8)

    yellow = cv2.morphologyEx(
        yellow, cv2.MORPH_CLOSE, kernel
    )

    white = cv2.morphologyEx(
        white, cv2.MORPH_CLOSE, kernel
    )

    def get_groups(mask, y):
        y0 = max(0, y - 4)
        y1 = min(h, y + 5)

        xs = np.where(
            np.count_nonzero(mask[y0:y1, :], axis=0) >= 2
        )[0]

        if len(xs) == 0:
            return []

        groups = []
        start = previous = int(xs[0])

        for value in xs[1:]:
            x = int(value)

            if x - previous > 6:
                width = previous - start + 1

                if 2 <= width <= min(160, int(0.25 * w)):
                    groups.append((start + previous) / 2.0)

                start = x

            previous = x

        width = previous - start + 1

        if 2 <= width <= min(160, int(0.25 * w)):
            groups.append((start + previous) / 2.0)

        return groups

    def fit_curve(points):
        if len(points) < 5:
            return None

        py = np.asarray(
            [point[0] for point in points], dtype=float
        )

        px = np.asarray(
            [point[1] for point in points], dtype=float
        )

        degree = 2 if (
            len(points) >= 7 and np.ptp(py) >= h * 0.12
        ) else 1

        try:
            fit = np.polyfit(py, px, degree)

            if degree == 1:
                fit = np.array([0.0, fit[0], fit[1]])

            for attempt in range(2):
                residual = np.abs(np.polyval(fit, py) - px)
                good = residual <= max(12.0, 0.035 * w)

                if np.count_nonzero(good) < 5:
                    break

                if np.count_nonzero(good) == len(py):
                    break

                gy = py[good]
                gx = px[good]

                degree2 = 2 if degree == 2 and len(gy) >= 6 else 1
                fit2 = np.polyfit(gy, gx, degree2)

                if degree2 == 1:
                    fit = np.array([0.0, fit2[0], fit2[1]])
                else:
                    fit = fit2

            return fit

        except (ValueError, np.linalg.LinAlgError):
            return None

    def choose_near(candidates, expected, max_distance):
        if not candidates or not np.isfinite(expected):
            return None

        if expected < -0.08 * w or expected > 1.08 * w:
            return None

        chosen = min(
            candidates,
            key=lambda x: abs(x - expected)
        )

        if abs(chosen - expected) > max_distance:
            return None

        return chosen

    def width_at_row(y):
        if lane_width <= 0.0:
            return 0.28 * w

        scale = (y - horizon_y) / max(
            1.0, target_y - horizon_y
        )

        scale = min(1.50, max(0.45, scale))

        return lane_width * scale

    def offset_from_boundary(fit, direction):
        # Offset the visible boundary to estimate the lane centre.
        result = fit.copy()
        result[2] += direction * lane_width / 2.0
        return result

    row_data = [
        (y, get_groups(white, y), get_groups(yellow, y))
        for y in rows
    ]

    white_fit = None
    yellow_fit = None

    # First frame: identify which white and yellow markings
    # enclose the lane nearest the vehicle's forward path.
    if not lane_locked:
        white_points = []
        yellow_points = []
        previous_pair = None

        for y, whites, yellows in row_data:
            if not whites or not yellows:
                continue

            best_pair = None
            best_score = float("inf")

            for wx in whites:
                for yx in yellows:
                    separation = abs(yx - wx)

                    if (
                        separation < 0.08 * w
                        or separation > 0.82 * w
                    ):
                        continue

                    middle = (wx + yx) / 2.0

                    if not (0.05 * w <= middle <= 0.95 * w):
                        continue

                    score = 0.80 * abs(middle - image_center)
                    score += 0.10 * abs(separation - 0.48 * w)

                    if previous_pair is not None:
                        prev_w, prev_y, prev_mid = previous_pair

                        score += 0.30 * abs(middle - prev_mid)
                        score += 0.10 * abs(wx - prev_w)
                        score += 0.10 * abs(yx - prev_y)

                    if score < best_score:
                        best_score = score
                        best_pair = (wx, yx, middle)

            if best_pair is not None:
                wx, yx, middle = best_pair

                white_points.append((y, wx))
                yellow_points.append((y, yx))

                previous_pair = (wx, yx, middle)

        white_fit = fit_curve(white_points)
        yellow_fit = fit_curve(yellow_points)

        if white_fit is None or yellow_fit is None:
            missing_center += 1
            return None

        wx = float(np.polyval(white_fit, target_y))
        yx = float(np.polyval(yellow_fit, target_y))

        separation = abs(yx - wx)
        center_x = (wx + yx) / 2.0

        if not (
            0 <= wx < w
            and 0 <= yx < w
            and 0 <= center_x < w
        ):
            missing_center += 1
            return None

        if not (0.08 * w <= separation <= 0.82 * w):
            missing_center += 1
            return None

        lane_side = 1 if yx > wx else -1
        lane_locked = True
        lane_width = separation
        base_lane_width = separation

    else:
        # Subsequent frames: follow the previously identified curves.
        white_points = []
        yellow_points = []

        max_jump = max(
            0.06 * w,
            0.16 * min(max(lane_width, 1.0),0.60 * w)
        )

        for y, whites, yellows in row_data:
            if last_white_fit is not None:
                white_expected = float(
                    np.polyval(last_white_fit, y)
                )

            elif last_center_fit is not None:
                white_expected = (
                    float(np.polyval(last_center_fit, y))
                    - lane_side * width_at_row(y) / 2.0
                )

            else:
                white_expected = image_center

            wx = choose_near(
                whites, white_expected, max_jump
            )

            if wx is not None:
                white_points.append((y, wx))
                divider_x = wx
            else:
                divider_x = white_expected

            if last_yellow_fit is not None:
                yellow_expected = float(
                    np.polyval(last_yellow_fit, y)
                )
            else:
                yellow_expected = (
                    divider_x + lane_side * width_at_row(y)
                )

            possible = []

            for yx in yellows:
                distance = lane_side * (yx - divider_x)

                # Yellow must stay on the established outer side.
                if distance <= 0:
                    continue

                expected_width = width_at_row(y)

                if not (
                    0.40 * expected_width
                    <= distance
                    <= 1.75 * expected_width
                ):
                    continue

                if abs(yx - yellow_expected) > max_jump:
                    continue

                possible.append(yx)

            if possible:
                yx = min(
                    possible,
                    key=lambda x: abs(x - yellow_expected)
                )

                yellow_points.append((y, yx))

        white_fit = fit_curve(white_points)
        yellow_fit = fit_curve(yellow_points)

        # Reject sudden jumps to a different visible marking.
        if white_fit is not None and last_white_fit is not None:
            old_x = float(
                np.polyval(last_white_fit, target_y)
            )

            new_x = float(
                np.polyval(white_fit, target_y)
            )

            if abs(new_x - old_x) > 0.18 * w:
                white_fit = None

        if yellow_fit is not None and last_yellow_fit is not None:
            old_x = float(
                np.polyval(last_yellow_fit, target_y)
            )

            new_x = float(
                np.polyval(yellow_fit, target_y)
            )

            if abs(new_x - old_x) > 0.18 * w:
                yellow_fit = None

    # Check the two curves at several road positions.
    pair_ok = False

    if white_fit is not None and yellow_fit is not None:
        test_rows = [
            int(h * r)
            for r in (0.44, 0.54, 0.64, 0.74, 0.82)
        ]

        target_white = float(
            np.polyval(white_fit, target_y)
        )

        target_yellow = float(
            np.polyval(yellow_fit, target_y)
        )

        target_separation = abs(target_yellow - target_white)
        width_ok = (
            base_lane_width <= 0.0
            or (
                0.85 * base_lane_width
                <= target_separation
                <= 1.10 * base_lane_width
            )
        )

        pair_ok = (
            0 <= target_white < w
            and 0 <= target_yellow < w
            and lane_side * (target_yellow - target_white) > 0
            and 0.06 * w <= target_separation <= 0.88 * w
            and width_ok
        )

        good_rows = 0

        for y in test_rows if pair_ok else []:
            wx = float(np.polyval(white_fit, y))
            yx = float(np.polyval(yellow_fit, y))

            if not (0 <= wx < w and 0 <= yx < w):
                continue

            separation = abs(yx - wx)

            if lane_side * (yx - wx) <= 0:
                pair_ok = False
                break

            if not (0.06 * w <= separation <= 0.88 * w):
                pair_ok = False
                break

            good_rows += 1

        if good_rows < 3:
            pair_ok = False

    center_fit = None
    recovered = False

    if pair_ok:
        # Both boundaries are valid, so use their midpoint.
        center_fit = (white_fit + yellow_fit) / 2.0

        wx = float(np.polyval(white_fit, target_y))
        yx = float(np.polyval(yellow_fit, target_y))

        measured_width = abs(yx - wx)

        if 0.08 * w <= measured_width <= 0.82 * w:
            if lane_width <= 0:
                lane_width = measured_width
            else:
                lane_width = (
                    0.95 * lane_width
                    + 0.05 * measured_width
                )

        last_white_fit = white_fit.copy()
        last_yellow_fit = yellow_fit.copy()

    elif lane_locked and lane_side != 0 and white_fit is not None:
        # Use the learned curve offset from the white divider.
        if white_center_offset is not None:
            center_fit = white_fit + white_center_offset
            recovered = True
        else:
            center_fit = offset_from_boundary(
                white_fit, lane_side
            )
            recovered = True

    elif lane_locked and lane_side != 0 and yellow_fit is not None:
        # Use the learned curve offset from the yellow boundary.
        if yellow_center_offset is not None:
            center_fit = yellow_fit + yellow_center_offset
            recovered = True
        else:
            center_fit = offset_from_boundary(
                yellow_fit, -lane_side
            )
            recovered = True

        recovered = True
        last_yellow_fit = yellow_fit.copy()

    if center_fit is None:
        missing_center += 1

        if last_center_fit is not None and missing_center <= 3:
            center_fit = last_center_fit.copy()
            recovered = True
        else:
            return None
    else:
        missing_center = 0

    raw_center = float(
        np.polyval(center_fit, target_y)
    )

    if not np.isfinite(raw_center):
        missing_center += 1
        return None

    # Never turn an invalid off-screen estimate into a valid-looking target.
    if not (0.0 <= raw_center < w):
        missing_center += 1

        if last_center_fit is None:
            return None

        center_fit = last_center_fit.copy()
        raw_center = float(
            np.polyval(center_fit, target_y)
        )
        recovered = True

        if not (
            np.isfinite(raw_center)
            and 0.0 <= raw_center < w
        ):
            return None

    center_x = raw_center

    if smooth_center is None:
        smooth_center = center_x
    else:
        alpha = 0.20 if recovered else 0.42

        smooth_center = (
            (1.0 - alpha) * smooth_center
            + alpha * center_x
        )

    # Estimate heading closer to the vehicle to reduce early turning.
    heading_y = h * 0.80

    slope = float(
        2.0 * center_fit[0] * heading_y
        + center_fit[1]
    )

    slope = float(np.clip(slope, -4.0, 4.0))
    raw_heading = float(np.arctan(slope))

    if smooth_heading is None:
        smooth_heading = raw_heading
    else:
        smooth_heading = (
            0.75 * smooth_heading
            + 0.25 * raw_heading
        )

    heading_error = smooth_heading
    # Positive steering is left. With image y increasing downward,
    # this slope convention gives a positive error for a leftward path.
    cross_track = float(image_center - smooth_center)
    # Learn boundary offsets only from a valid pair.
    if (
        pair_ok
        and center_fit is not None
        and white_fit is not None
        and yellow_fit is not None
    ):
        last_center_fit = center_fit.copy()

        new_white_offset = center_fit - white_fit
        new_yellow_offset = center_fit - yellow_fit

        if white_center_offset is None:
            white_center_offset = new_white_offset.copy()
            yellow_center_offset = new_yellow_offset.copy()
        else:
            white_center_offset = (
                0.85 * white_center_offset
                + 0.15 * new_white_offset
            )

            yellow_center_offset = (
                0.85 * yellow_center_offset
                + 0.15 * new_yellow_offset
            )

    detection = {
        "center_x": float(smooth_center),
        "cross_track_error_px": cross_track,
        "heading_error": heading_error,
        "lane": "right" if lane_side == 1 else "left",
        "lane_width_px": max(lane_width, 0.12 * w),
        "recovered": recovered,
    }

    last_detection = detection
    print(
        f"[FOLLOW] center={smooth_center:.1f}, "
        f"cte={cross_track:.1f}, "
        f"heading={heading_error:.3f}, "
        f"width={lane_width:.1f}, "
        f"recovered={recovered}"
    )
    return detection


def compute_control(detection, stop_requested):
    '''
    Purpose:
    ---
    Turn the errors from detect_lane() into the three numbers the vehicle is
    driven with, and obey the stop flag.

    This is where your Task 1B controller and your Task 1A steering geometry
    meet: work out one steering angle from the error, then split it into the
    two front-wheel angles the joints actually take.

    Input Arguments:
    ---
    `detection` :   [ any ]
        Whatever detect_lane() returned for this frame.

    `stop_requested` :  [ bool ]
        True while the evaluation is holding the vehicle at a stop point.
        This is a LEVEL, not a pulse: it stays True for the whole duration of
        the stop and goes False when the wait is over. While it is True the
        vehicle must be STATIONARY, so return a speed of 0.0. There is no
        stopwatch for you to write and no duration to hard-code.

    Returns:
    ---
    `left_angle`, `right_angle` :   [ float ]
        The two front-wheel steering angles, in RADIANS, written straight to
        the steering joints. The scene's steering joints are POSITIVE TO THE
        LEFT. Nothing is clamped for you - the vehicle's own limit is 45
        degrees, and commanding past it will not steer any harder.

    `speed` :   [ float ]
        Target velocity for both drive motors, in rad/s. 0.0 holds the
        vehicle still.

    Example call:
    ---
    left_angle, right_angle, speed = compute_control(detection, stop_requested)

    REMEMBER:
    ---
    Keep the loop running while stopped - set the SPEED to zero, not your
    program. If you stop reading frames you will not see the flag drop.

    Sooner or later a frame comes back with nothing usable in it. Holding the
    last steering angle, easing off the throttle and stopping dead are all
    defensible - doing nothing is not.
    '''

    # ADD YOUR CODE HERE
    import math

    global last_steering, last_speed, last_detection, control_count

    # Always obey the evaluator's stop signal.
    if stop_requested:
        last_steering = 0.0
        last_speed = 0.0
        return 0.0, 0.0, 0.0

    if detection is None:
        if last_detection is not None and missing_center <= 2:
            steering = last_steering
            speed = 0.10
        else:
            last_steering = 0.0
            last_speed = 0.0
            return 0.0, 0.0, 0.0

    else:
        last_detection = detection

        error = (
            detection["cross_track_error_px"]
            / max(detection["lane_width_px"], 1.0)
        )

        heading = detection["heading_error"]

        # Small errors should not cause continual steering corrections.
        if abs(error) < 0.025:
            error = 0.0

        # Stanley-style steering: align with the road, then correct offset.
        correction = math.atan(
            0.90 * error / (0.40 + max(last_speed, 0.0))
        )

        desired = 0.30 * heading + correction
        desired = max(-0.58, min(0.58, desired))

        # Smooth steering without forcing a turn in either direction.
        steering = (
            0.68 * last_steering
            + 0.32 * desired
        )

        # Limit the change in either direction.
        steering = max(
            last_steering - 0.05,
            min(last_steering + 0.05, steering)
        )

        steering = max(-0.58, min(0.58, steering))
        last_steering = steering

        if detection.get("recovered", False):
            speed = 0.16
        elif abs(heading) > 0.45 or abs(desired) > 0.45:
            speed = 0.18
        elif abs(heading) > 0.28 or abs(desired) > 0.30:
            speed = 0.24
        elif abs(heading) > 0.15 or abs(desired) > 0.18:
            speed = 0.32
        else:
            speed = 0.42

    # Ackermann steering geometry.
    # Positive angles turn left; the inside wheel gets the larger angle.
    wheelbase = 0.120
    track = 0.055

    if abs(steering) < 0.001:
        left_angle = 0.0
        right_angle = 0.0

    else:
        radius = wheelbase / abs(math.tan(steering))

        inner = math.atan(
            wheelbase / max(0.01, radius - track / 2.0)
        )

        outer = math.atan(
            wheelbase / (radius + track / 2.0)
        )

        if steering > 0:
            left_angle = inner
            right_angle = outer
        else:
            left_angle = -outer
            right_angle = -inner

    last_speed = speed

    if detection is None:
        print(
            f"[CTRL] detection=None, "
            f"missing={missing_center}, speed={speed:.2f}"
        )
    elif abs(heading) > 0.15 or abs(error) > 0.04:
        print(
            f"[TURN] lane={detection['lane']}, "
            f"cte={detection['cross_track_error_px']:.1f}, "
            f"heading={heading:.3f}, "
            f"desired={desired:.3f}, "
            f"steering={steering:.3f}, "
            f"left={left_angle:.3f}, "
            f"right={right_angle:.3f}, "
            f"speed={speed:.2f}"
        )
    if control_count < 25:
        if detection is not None:
            print(
                f"[CTRL] cte={detection['cross_track_error_px']:.1f}, "
                f"heading={detection['heading_error']:.3f}, "
                f"desired={desired:.3f}, "
                f"steering={steering:.3f}, "
                f"speed={speed:.2f}"
            )
        else:
            print(
                f"[CTRL] detection=None, "
                f"missing={missing_center}, speed={speed:.2f}"
            )

        control_count += 1

    return (float(left_angle),float(right_angle),float(speed))


##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
