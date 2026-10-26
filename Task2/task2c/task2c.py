'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2C of Niti Vahan (NV) Theme.
*
*  Filename:         task2c.py
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
# Author List:      Dev Patel, Priyansh Miyani
# Filename:         task2c.py
# Functions:        parking_manoeuvre
# Global variables: _last_steering, _lost_frames, _parked_frames _last_pose, _progress_count, _crossing_done, _align_direction


####################### IMPORT MODULES #######################
import argparse
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2C.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# Two parking bays sit side by side to the RIGHT of the road, long axis along
# it. Each carries an ArUco marker at its head, facing back down the road:
# id 0 on bay 0 (nearer the road), id 1 on bay 1 (further right). The vehicle
# is spawned on the road, parallel to the bays, short of them and to their left.
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Which bay to park in: 0 or 1. This is the default - `python task2c.py --bay 1`
# overrides it for one run without editing the file.
TARGET_ID = 1

# Bay geometry, for your parking logic. Distances are measured along the road
# from the vehicle's centre to the line the markers stand on.
BAY_MOUTH_FORWARD  = 0.37    # m - the vehicle's centre is at the bay mouth here
BAY_MIDDLE_FORWARD = 0.19    # m - ...and in the middle of the bay here
BAY_WIDTH          = 0.22    # m - the vehicle is 0.18 m across
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################
_last_steering = 0.0 
_lost_frames=0
_parked_frames=0
_last_pose=None
_progress_count=0
_crossing_done=False
_align_direction=1.0
def parking_manoeuvre(pose):
    '''
    Purpose:
    ---
    Decide, every frame, how to steer and how fast to drive so that the
    vehicle ends up inside the target bay - on its centre line, parallel to
    it, stopped in the middle.

    This is the whole of Task 2C. Finding the marker in the camera image and
    working out where the vehicle is relative to the bay is done for you
    below; you are handed the answer as `pose` and decide what to do with it.

    Input Arguments:
    ---
    `pose` :    [ dict or None ]
        None on a frame where the target bay's marker was not seen.
        Otherwise where the vehicle is, relative to the target bay:

          'lateral'  [float] metres. Sideways offset of the front axle from
                             the bay's centre line. POSITIVE = the vehicle
                             is to the LEFT of the line. 0 = on it.
          'heading'  [float] radians. Direction the vehicle points, relative
                             to the bay's long axis. 0 = parallel to the bay.
                             POSITIVE = turned to the LEFT.
          'forward'  [float] metres. Distance along the road from the
                             vehicle's centre to the marker line. It shrinks
                             as you approach: BAY_MOUTH_FORWARD at the mouth,
                             BAY_MIDDLE_FORWARD in the middle of the bay.
          'bearing'  [float] radians. Where the marker appears in the camera
                             view. 0 = straight ahead, POSITIVE = to the LEFT.

    Returns:
    ---
    `steering` :    [ float ]
        Steering angle in RADIANS, POSITIVE TO THE LEFT. It is split into
        the two front-wheel angles and clamped to the vehicle's 45 degree
        limit for you.

    `speed` :       [ float ]
        Target velocity for both drive motors, in rad/s. Positive drives
        forwards. 0.0 holds the vehicle still.

    `parked` :      [ bool ]
        True once the vehicle is parked. Returning True stops the vehicle
        and ends the run, so return it only when you are done.

    Example call:
    ---
    steering, speed, parked = parking_manoeuvre(pose)

    REMEMBER:
    ---
    Steering straight AT the marker is not parking. The bays are to the
    right of the road, so pointing at the marker brings the vehicle in on a
    diagonal, and a car that arrives crooked does not fit a bay with ~20 mm
    to spare each side. Both 'lateral' and 'heading' have to reach zero.

    The camera is 95 degrees wide. Turn much more than ~40 degrees across the
    road and the marker slides out of view - `pose` becomes None, and there
    is nothing left to steer by.

    A single None frame is normal. Decide what to do on one; doing nothing
    sensible is how a vehicle drives off.
    '''

    # ADD YOUR CODE HERE
    global _last_steering, _lost_frames, _parked_frames
    global _last_pose, _progress_count, _crossing_done, _align_direction

    # A brief missing detection is normal; do not let one frame change the path.
    if pose is None:
        _lost_frames += 1
        _parked_frames = 0
        if _lost_frames <= 2 and _last_pose is not None:
            return float(_last_steering), 0.04, False
        _last_steering = 0.0
        return 0.0, 0.0, False

    _lost_frames = 0
    try:
        lateral = float(pose["lateral"])   # positive = vehicle LEFT of bay centreline
        heading = float(pose["heading"])   # positive = nose points LEFT
        forward = float(pose["forward"])
        bearing = float(pose["bearing"])
    except (KeyError, TypeError, ValueError):
        _parked_frames = 0
        return 0.0, 0.0, False

    if not all(np.isfinite(v) for v in (lateral, heading, forward, bearing)):
        _parked_frames = 0
        return 0.0, 0.0, False

    _last_pose = {"lateral": lateral, "heading": heading, "forward": forward, "bearing": bearing}
    _progress_count += 1
    far_bay = int(TARGET_ID) == 1
    depth_error = forward - float(BAY_MIDDLE_FORWARD)

    # Reference crossing: Bay 0 ~= 25 degrees; Bay 1 ~= 38.5 degrees.
    # Once the vehicle reaches the centreline (lateral <= 8 mm), latch the
    # crossing complete. Do not re-enter the fixed-angle crossing phase due to
    # a noisy positive/negative lateral estimate on following frames.
    crossing_heading = np.deg2rad(38.5 if far_bay else 25.0)
    cross_deadline = float(BAY_MOUTH_FORWARD) + (0.018 if far_bay else 0.045)
    if lateral <= 0.008:
        _crossing_done = True

    if not _crossing_done and lateral > 0.008 and forward > cross_deadline:
        phase = "CROSS"
        desired_heading = -crossing_heading
        heading_gain = 1.55 if not far_bay else 1.45
        lateral_gain = 0.20
        steering_limit = np.deg2rad(29.0 if not far_bay else 37.0)
        max_delta = np.deg2rad(13.0 if not far_bay else 15.0)
    elif not _crossing_done:
        # If the target line has not been reached by the preferred deadline,
        # keep correcting towards a short lookahead point on the centreline.
        phase = "CENTER"
        lookahead = 0.25 if not far_bay else 0.29
        desired_heading = -float(np.arctan2(lateral, lookahead))
        heading_gain = 1.65
        lateral_gain = 0.32
        steering_limit = np.deg2rad(26.0 if not far_bay else 29.0)
        max_delta = np.deg2rad(11.0)
    else:
        # Use a short lookahead while lateral error is appreciable to move the
        # front axle onto the centreline. Once close to the line, lengthen the
        # lookahead so the vehicle straightens without weaving.
        phase = "ALIGN"
        abs_lateral = abs(lateral)
        if abs_lateral > 0.010:
            lookahead = 0.14
        elif abs_lateral > 0.006:
            lookahead = 0.25
        else:
            lookahead = 0.48
        desired_heading = -float(np.arctan2(lateral, lookahead))
        heading_gain = 2.00
        lateral_gain = 0.35
        steering_limit = np.deg2rad(29.0 if not far_bay else 33.0)
        max_delta = np.deg2rad(14.0 if not far_bay else 15.0)

    heading_error = desired_heading - heading
    target_steering = heading_gain * heading_error - lateral_gain * lateral
    if phase == "CROSS" and forward > BAY_MOUTH_FORWARD + 0.20:
        target_steering += 0.02 * bearing
    target_steering = float(np.clip(target_steering, -steering_limit, steering_limit))

    filtered = 0.25 * float(_last_steering) + 0.75 * target_steering
    steering_forward = float(np.clip(
        filtered,
        float(_last_steering) - max_delta,
        float(_last_steering) + max_delta,
    ))
    steering_forward = float(np.clip(steering_forward, -steering_limit, steering_limit))

    # The target is the bay middle, and parking requires all three coordinates
    # to agree simultaneously for several consecutive frames.
    # Tight but attainable envelope; Bay 1 allows slightly more heading error
    # because of its longer, harder crossing. All three must pass at once.
    lateral_tol = 0.006 if not far_bay else 0.011
    heading_tol = np.deg2rad(1.5 if not far_bay else 2.5)
    depth_tol = 0.012
    lateral_ok = abs(lateral) <= lateral_tol
    heading_ok = abs(heading) <= heading_tol
    depth_ok = abs(depth_error) <= depth_tol
    if lateral_ok and heading_ok and depth_ok:
        _parked_frames += 1
        _last_steering = 0.0
        if _parked_frames >= 4:
            print(f"[PARK] parked: lat={lateral:+.3f}m, heading={np.rad2deg(heading):+.2f}deg, forward={forward:.3f}m")
            return 0.0, 0.0, True
        return 0.0, 0.0, False
    _parked_frames = 0

    # Use useful speed during the long approach, then taper near the middle.
    # The previous speed schedule spent most of the 120 s budget creeping at
    # 0.045-0.10 rad/s, despite still being short of the target.
    if phase == "CROSS":
        speed = 0.95 if forward > BAY_MOUTH_FORWARD + 0.10 else 0.70
    elif phase == "CENTER":
        if forward > BAY_MOUTH_FORWARD + 0.15:
            speed = 0.95
        elif forward > BAY_MOUTH_FORWARD + 0.06:
            speed = 0.70
        elif forward > BAY_MIDDLE_FORWARD + 0.05:
            speed = 0.42
        elif depth_error > depth_tol:
            speed = min(0.25, max(0.16, 1.2 * depth_error))
        else:
            speed = -0.15 if depth_error < -depth_tol else 0.14
    else:  # ALIGN
        if forward > BAY_MOUTH_FORWARD + 0.20:
            speed = 1.00
        elif forward > BAY_MOUTH_FORWARD + 0.10:
            speed = 0.80
        elif forward > BAY_MOUTH_FORWARD:
            speed = 0.60
        elif forward > BAY_MIDDLE_FORWARD + 0.06:
            speed = 0.38
        elif forward > BAY_MIDDLE_FORWARD + 0.025:
            speed = 0.26
        elif depth_error > depth_tol:
            speed = min(0.24, max(0.16, 2.0 * depth_error))
        elif depth_error < -depth_tol:
            speed = max(-0.22, min(-0.15, 2.0 * depth_error))
        elif lateral_ok and heading_ok:
            speed = 0.0
        else:
            if depth_error > 0.010:
                _align_direction = 1.0
            elif depth_error < -0.010:
                _align_direction = -1.0
            speed = 0.16 * _align_direction

    steering = -steering_forward if speed < 0.0 else steering_forward
    _last_steering = steering_forward

    if _progress_count % 40 == 0:
        print(f"[PARK] bay={TARGET_ID} phase={phase} lat={lateral:+.3f}m head={np.rad2deg(heading):+.1f}deg forward={forward:.3f}m speed={speed:+.3f} crossed={_crossing_done}")
    return float(steering), float(speed), False



##############################################################
############## END OF YOUR IMPLEMENTATION ####################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera, finding the marker, working
# out the vehicle's pose relative to the bay, and writing to the joints. None
# of it involves a decision you are marked on.
#
#####################################################################

WHEELBASE      = 0.120     # L, metres
TRACK_WIDTH    = 0.110     # W, metres
WHEEL_OFFSET   = 0.0275    # O, metres
STEER_LIMIT    = np.deg2rad(45.0)

# ---- camera and marker, as in Task_2C.ttt ----
IMG_W, IMG_H  = 640, 480
FOCAL_PX      = (IMG_W / 2) / np.tan(np.deg2rad(95.0) / 2)   # 95 deg horizontal FOV
CAMERA_MATRIX = np.array([[FOCAL_PX, 0, IMG_W / 2],
                          [0, FOCAL_PX, IMG_H / 2],
                          [0, 0, 1]])
CAM_PITCH     = np.deg2rad(30.0)     # camera tilted 30 deg down
CAM_ABOVE_MK  = 0.176 - 0.110        # camera height minus marker-centre height, m
CAM_AHEAD     = 0.047                # camera is this far ahead of the vehicle centre
# The black pattern, not the board: the 0.12 m board has a white border and
# the pattern is 80% of it.
MARKER_SIDE   = 0.096
_H = MARKER_SIDE / 2
MARKER_CORNERS_3D = np.array([[-_H, _H, 0], [_H, _H, 0],
                              [_H, -_H, 0], [-_H, -_H, 0]], np.float32)

# ---- pose filter ----
YAW_BLEND       = 0.12               # how far each marker reading pulls the heading
YAW_GATE        = np.deg2rad(15.0)   # readings further than this from the prediction are rejected
AMBIGUITY_RATIO = 1.5                # PnP's two answers must differ this much in fit to trust one


def ackermann_wheel_angles(delta):
    '''Split one steering angle into the two front-wheel angles.

    This is Task 1A's geometry, provided here so that Task 2C is only about
    the manoeuvre. The kingpins sit WHEEL_OFFSET inboard of each wheel
    centre, so the triangle is built on the kingpin-to-kingpin width.
    '''
    if abs(delta) < 1e-9:
        return 0.0, 0.0
    l = WHEELBASE
    w = TRACK_WIDTH - 2 * WHEEL_OFFSET
    s, c = np.sin(delta), np.cos(delta)
    left = np.arctan2(2 * l * s, 2 * l * c - w * s)
    right = np.arctan2(2 * l * s, 2 * l * c + w * s)
    return float(left), float(right)


def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.'''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2C.ttt open?\n'
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
                f'  Open Task_2C.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    Three corrections turn the sensor's raw buffer into an image, and none of
    the three announces itself if you leave it out:

      * handleVisionSensor() - the sensor renders only when asked.
      * cv2.flip(frame, 0)   - the buffer's first row is the BOTTOM row.
      * RGB2BGR              - the buffer is RGB; OpenCV assumes BGR.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def find_marker(detector, frame, target_id):
    '''The target marker's four corners in the image (top-left, top-right,
    bottom-right, bottom-left), or None when it is not in this frame.'''
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    for quad, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) == int(target_id):
            return quad.reshape(4, 2)
    return None


class PoseEstimator:
    '''Where the vehicle is relative to the target bay, from the marker.

    Position comes straight from the marker each frame. Heading is harder: a
    small marker far away gives a noisy reading, and a square seen nearly
    face-on has two mirror-image poses that fit its corners almost equally
    well. So heading is predicted from the steering actually commanded, and
    each clear marker reading pulls that prediction a little toward itself.
    The vehicle is spawned parallel to the bays, which is where it starts.
    '''

    def __init__(self):
        self.heading = 0.0      # rad, + = turned RIGHT (internal convention)
        self.rho = None         # smoothed horizontal range to the marker, m

    @staticmethod
    def _bearing(corners):
        '''Horizontal bearing to the marker centre, rad, + = to the RIGHT.'''
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        xc = (cx - IMG_W / 2) / FOCAL_PX
        yc = (IMG_H / 2 - cy) / FOCAL_PX
        return float(np.arctan2(xc, np.cos(CAM_PITCH) + yc * np.sin(CAM_PITCH)))

    @staticmethod
    def _pnp(corners):
        '''Heading (rad, + = RIGHT, or None if ambiguous) and horizontal range (m).'''
        _, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            MARKER_CORNERS_3D, corners.astype(np.float32), CAMERA_MATRIX,
            np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = []
        for rv, tv, e in zip(rvecs, tvecs, np.array(errs).flatten()):
            R, _ = cv2.Rodrigues(rv)
            fwd = R.T @ np.array([0.0, 0.0, 1.0])   # camera forward, in the marker frame
            sols.append((float(e), float(np.arctan2(fwd[0], -fwd[2])), tv.flatten()))
        sols.sort(key=lambda s: s[0])
        t = sols[0][2]
        rho = float(np.sqrt(max(t @ t - CAM_ABOVE_MK ** 2, 1e-6)))
        clear = len(sols) < 2 or sols[1][0] > AMBIGUITY_RATIO * max(sols[0][0], 1e-6)
        return (sols[0][1] if clear else None), rho

    def update(self, corners, last_steering):
        '''The pose handed to parking_manoeuvre(), or None if no marker.'''
        if corners is None:
            return None
        bearing = self._bearing(corners)
        yaw_seen, rho_now = self._pnp(corners)
        if self.rho is None:
            self.rho = rho_now
        rho_prev, self.rho = self.rho, 0.6 * self.rho + 0.4 * rho_now
        # Distance covered since the last frame: range shrinks at cos(bearing)
        # per metre travelled.
        ds = min(max((rho_prev - self.rho) / max(np.cos(bearing), 0.3), 0.0), 0.01)
        # Steering LEFT (+) turns the heading LEFT, NEGATIVE in this convention.
        self.heading -= np.tan(last_steering) / WHEELBASE * ds
        if yaw_seen is not None and abs(yaw_seen - self.heading) < YAW_GATE:
            self.heading += YAW_BLEND * (yaw_seen - self.heading)

        los = bearing + self.heading                # line of sight, from the bay axis
        lateral_cam = -self.rho * np.sin(los)       # + = RIGHT of the centre line
        forward_cam = self.rho * np.cos(los)
        lateral_front = lateral_cam + (WHEELBASE / 2 - CAM_AHEAD) * np.sin(self.heading)
        forward = forward_cam + CAM_AHEAD * np.cos(self.heading)
        # Handed over in the Task 1 convention: POSITIVE = LEFT throughout.
        return {
            'lateral': float(-lateral_front),
            'heading': float(-self.heading),
            'forward': float(forward),
            'bearing': float(-bearing),
        }


def drive(sim, handles, steering, speed):
    '''Split the steering and write it, with the drive speed.'''
    left, right = ackermann_wheel_angles(steering)
    sim.setJointTargetPosition(handles['steer_left'], left)
    sim.setJointTargetPosition(handles['steer_right'], right)
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def parse_args():
    '''The bay to park in, from the command line - or TARGET_ID if not given.'''
    parser = argparse.ArgumentParser(
        description='Task 2C - park the vehicle in one of the two bays.')
    parser.add_argument('--bay', type=int, choices=(0, 1), default=None,
                        help='which bay to park in (default: TARGET_ID in this file)')
    return parser.parse_args()


def main():
    global TARGET_ID
    # Read the flag before touching the simulator, so --help and a bad
    # value are answered straight away.
    args = parse_args()
    if args.bay is not None:
        TARGET_ID = args.bay

    sim, handles = connect_and_locate()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters())
    estimator = PoseEstimator()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, spawns the vehicle and starts the simulation,
    # so wait for that rather than starting your own.
    deadline = time.monotonic() + 120.0
    if sim.getSimulationState() != sim.simulation_advancing_running:
        print('Waiting for the simulation to start - press Play in CoppeliaSim.')
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - parking in bay %d.' % TARGET_ID)

    frames, seen, parked, steering = 0, 0, False, 0.0
    try:
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            corners = find_marker(detector, frame, TARGET_ID)
            if corners is not None:
                seen += 1
            pose = estimator.update(corners, steering)

            steering, speed, parked = parking_manoeuvre(pose)
            if parked:
                drive(sim, handles, 0.0, 0.0)
                print('Parked after %d frames.' % frames)
                break
            steering = float(np.clip(steering, -STEER_LIMIT, STEER_LIMIT))
            drive(sim, handles, steering, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        try:
            drive(sim, handles, 0.0, 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed, marker {TARGET_ID} seen in {seen}.')
    if not parked:
        print('The run ended without parking_manoeuvre() returning parked=True.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
