"""
Minimal agent for log-replay mode.

This agent does nothing except return neutral controls. When CUSTOM_EGO_LOG_REPLAY=1
is set, the ego vehicle's actual movement is controlled by LogReplayFollower in
route_scenario.py, which uses set_transform() to follow the logged trajectory.

This agent exists solely to satisfy the leaderboard's agent requirement without
performing any complex initialization that could fail on custom maps.
"""

import datetime
import os
import pathlib

import carla
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def get_entry_point():
    return "LogReplayAgent"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer environment variable with default and minimum."""
    try:
        val = int(os.environ.get(name, default))
        return max(minimum, val)
    except (TypeError, ValueError):
        return default


class LogReplayAgent(AutonomousAgent):
    """Minimal agent that returns neutral controls for log-replay mode."""

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        """Set up log-replay agent with image capture support."""
        self.track = Track.SENSORS
        self._ego_vehicles_num = ego_vehicles_num
        self.ego_vehicles_num = ego_vehicles_num  # Required by AgentWrapper
        
        # Image capture settings (same pattern as tcp_agent.py)
        self.save_path = None
        self.logreplayimages_path = None
        self.calibration_path = None
        self.save_interval = _env_int("TCP_SAVE_INTERVAL", 1, minimum=1)  # Save every frame by default
        self.capture_sensor_frames = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
        self.capture_logreplay_images = os.environ.get("TCP_CAPTURE_LOGREPLAY_IMAGES", "").lower() in ("1", "true", "yes")
        self._frame_count = 0
        self._saved_frames = 0
        
        # Set up save paths if SAVE_PATH is defined
        save_path_env = os.environ.get("SAVE_PATH")
        if save_path_env:
            now = datetime.datetime.now()
            routes_env = os.environ.get("ROUTES", "logreplay")
            string = pathlib.Path(routes_env).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
            
            self.save_path = pathlib.Path(save_path_env) / string
            try:
                self.save_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(f"[LogReplayAgent] Warning: Could not create save_path: {exc}")
                self.save_path = None
            
            if self.save_path:
                self.logreplayimages_path = self.save_path / "logreplayimages"
                if self.capture_logreplay_images or self.capture_sensor_frames:
                    try:
                        self.logreplayimages_path.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(f"[LogReplayAgent] Warning: Could not create logreplayimages_path: {exc}")
                # Backward-compatible alias
                self.calibration_path = self.logreplayimages_path
                
                # Create directories for each ego vehicle (only rgb_front)
                for ego_id in range(self.ego_vehicles_num):
                    try:
                        (self.save_path / f'rgb_{ego_id}').mkdir(parents=True, exist_ok=True)
                        (self.save_path / f'meta_{ego_id}').mkdir(parents=True, exist_ok=True)
                        if self.capture_logreplay_images and self.logreplayimages_path:
                            # Only create rgb_front_n directories
                            (self.logreplayimages_path / f'rgb_front_{ego_id}').mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        print(f"[LogReplayAgent] Warning: Could not create dirs for ego {ego_id}: {exc}")
                
                print(f"[LogReplayAgent] Image capture enabled:")
                print(f"  save_path: {self.save_path}")
                print(f"  logreplayimages_path: {self.logreplayimages_path}")
                print(f"  capture_logreplay_images: {self.capture_logreplay_images}")
                print(f"  capture_sensor_frames: {self.capture_sensor_frames}")
                print(f"  save_interval: {self.save_interval}")
                if not HAS_CV2:
                    print(f"  WARNING: cv2 not available, images will not be saved!")

    def sensors(self):
        """Return sensor setup - only front RGB camera for each ego vehicle."""
        return [
            # Front view camera (rgb_front_0, rgb_front_1, etc. for each ego)
            {
                "type": "sensor.camera.rgb",
                "id": "rgb_front",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1280,
                "height": 720,
                "fov": 100,
            },
        ]

    def run_step(self, input_data, timestamp):
        """Return neutral controls for all ego vehicles; save images if enabled."""
        self._frame_count += 1
        
        # Save images if capture is enabled and this is a save frame
        should_save = (
            self.capture_logreplay_images 
            and self.logreplayimages_path is not None 
            and HAS_CV2
            and (self._frame_count % self.save_interval == 0)
        )
        
        if should_save:
            self._save_sensor_images(input_data, timestamp)
        
        # Must return a list of controls, one per ego vehicle
        control_all = []
        for _ in range(self.ego_vehicles_num):
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            control_all.append(control)
        return control_all

    def _save_sensor_images(self, input_data, timestamp):
        """Save rgb_front sensor images from input_data to disk."""
        if not self.logreplayimages_path or not HAS_CV2:
            return
        
        frame_id = self._saved_frames
        self._saved_frames += 1
        
        # Only save rgb_front for each ego vehicle
        for ego_id in range(self.ego_vehicles_num):
            # Sensor tag format: rgb_front_<ego_id>
            sensor_tag = f"rgb_front_{ego_id}"
            
            if sensor_tag in input_data:
                try:
                    # input_data[sensor_tag] is a tuple: (frame_number, image_data)
                    frame_num, image_data = input_data[sensor_tag]
                    
                    # Convert to numpy array if needed
                    if hasattr(image_data, 'shape'):
                        img = image_data
                    else:
                        # It's raw bytes, need to decode
                        continue
                    
                    # Handle different image formats
                    if len(img.shape) == 3:
                        if img.shape[2] == 4:
                            # BGRA -> BGR
                            img = img[:, :, :3]
                        elif img.shape[2] == 3:
                            # Already RGB or BGR
                            pass
                    
                    # Save image
                    out_dir = self.logreplayimages_path / f'rgb_front_{ego_id}'
                    out_path = out_dir / f'{frame_id:06d}.jpg'
                    cv2.imwrite(str(out_path), img)
                except Exception as exc:
                    # Don't spam errors, just log occasionally
                    if frame_id == 0:
                        print(f"[LogReplayAgent] Warning: Failed to save {sensor_tag}: {exc}")

    def destroy(self):
        """Cleanup."""
        if self._saved_frames > 0:
            print(f"[LogReplayAgent] Saved {self._saved_frames} frames to {self.logreplayimages_path}")
