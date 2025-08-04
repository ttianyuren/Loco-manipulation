import os
os.environ['MUJOCO_GL'] = 'osmesa'

import asyncio
import json
import websockets
import threading
import time
import base64
import io
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np
from PIL import Image
import queue
import statistics
from collections import deque

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_HERE = Path(__file__).parent
_XML = _HERE.parent / "sim" / "models" / "pal_tiago_dual" / "tiago_scene.xml"

_JOINT_NAMES = [
    "1_joint", "2_joint", "3_joint", "4_joint", 
    "5_joint", "6_joint", "7_joint",
]
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}

class PerformanceMonitor:
    def __init__(self):
        self.latencies = deque(maxlen=100)
        self.render_times = deque(maxlen=100)
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def add_latency(self, latency):
        self.latencies.append(latency)
        
    def add_render_time(self, render_time):
        self.render_times.append(render_time)
        
    def get_stats(self):
        return {
            'avg_latency': statistics.mean(self.latencies) if self.latencies else 0,
            'avg_render_time': statistics.mean(self.render_times) if self.render_times else 0,
            'fps': self.fps_counter / (time.time() - self.fps_start_time) if self.fps_counter > 0 else 0
        }

class WebTiagoServer:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(_XML))
        self.data = mujoco.MjData(self.model)
        
        # Initialize renderer with OSMesa
        try:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            print("Renderer initialized successfully")
        except Exception as e:
            print(f"Renderer initialization failed: {e}")
            raise
        
        self.setup_robot()
        
        # WebSocket and control state
        self.clients = set()
        self.command_queue = queue.Queue()
        self.current_mocap_index = 0
        self.step_size = 0.02
        self.running = True
        
        # Frame broadcasting - simplified
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
    def setup_robot(self):
        """Initialize robot configuration and tasks"""
        joint_names: List[str] = []
        velocity_limits: dict[str, float] = {}
        for prefix in ["left", "right"]:
            for n in _JOINT_NAMES:
                name = f"arm_{prefix}_{n}"
                joint_names.append(name)
                velocity_limits[name] = _VELOCITY_LIMITS[n]
        joint_names.extend(["base_x","base_y","base_th","torso_lift_joint"])
        
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(f"{name}_position").id for name in joint_names])
        self.configuration = mink.Configuration(self.model)
        
        # Setup tasks
        self.base_task = mink.FrameTask(
            frame_name="base_link", frame_type="body",
            position_cost=1.0, orientation_cost=1.0,
        )
        self.l_ee_task = mink.FrameTask(
            frame_name="left_gripper", frame_type="site",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
        )
        self.r_ee_task = mink.FrameTask(
            frame_name="right_gripper", frame_type="site",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-1)
        
        self.tasks = [self.base_task, self.l_ee_task, self.r_ee_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, velocity_limits),
        ]
        
        self.mocap_ids = [
            self.model.body("base_target").mocapid[0],
            self.model.body("left_gripper_target").mocapid[0], 
            self.model.body("right_gripper_target").mocapid[0]
        ]
        
        # Initialize robot
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("neutral_pose").id)
        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.base_task.set_target_from_configuration(self.configuration)
        
        mink.move_mocap_to_frame(self.model, self.data, "left_gripper_target", "left_gripper", "site")
        mink.move_mocap_to_frame(self.model, self.data, "right_gripper_target", "right_gripper", "site")
        
    def get_frame_as_base64(self):
        """Render current frame and return as base64 encoded image"""
        try:
            start_time = time.time()
            
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            
            # Flip vertically (MuJoCo renders upside down)
            pixels = np.flipud(pixels)
            
            img = Image.fromarray(pixels)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            render_time = (time.time() - start_time) * 1000  # Convert to ms
            self.perf_monitor.add_render_time(render_time)
            self.perf_monitor.fps_counter += 1
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Rendering error: {e}")
            return None
    
    def compensate_gravity(self, subtree_ids):
        """Apply gravity compensation"""
        self.data.qfrc_applied[:] = 0.0
        jac = np.empty((3, self.model.nv))
        for subtree_id in subtree_ids:
            total_mass = self.model.body_subtreemass[subtree_id]
            mujoco.mj_jacSubtreeCom(self.model, self.data, jac, subtree_id)
            self.data.qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac
    
    def simulation_loop(self):
        """Main simulation loop"""
        rate = RateLimiter(frequency=200.0, warn=False)
        lift_subtree_id = self.model.body("torso_lift_link").id
        frame_counter = 0
        last_perf_report = time.time()
        
        print("Starting simulation loop...")
        
        while self.running:
            # Process commands
            try:
                while True:
                    command = self.command_queue.get_nowait()
                    self.handle_control_command(command['command'], command['direction'])
            except queue.Empty:
                pass
            
            # Update task targets
            base_pose = self.data.mocap_pos[self.mocap_ids[0]].copy()
            base_pose[2] = 0
            self.data.mocap_pos[self.mocap_ids[0]] = base_pose
            
            self.base_task.set_target(mink.SE3.from_mocap_id(self.data, self.mocap_ids[0]))
            self.l_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.mocap_ids[1]))
            self.r_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.mocap_ids[2]))
            
            # Solve IK
            vel = mink.solve_ik(
                self.configuration, self.tasks, rate.dt, "daqp",
                limits=self.limits, damping=1e-5,
            )
            self.configuration.integrate_inplace(vel, rate.dt)
            
            # Apply to simulation
            self.data.ctrl[self.actuator_ids] = self.configuration.q[self.dof_ids]
            self.compensate_gravity([lift_subtree_id])
            mujoco.mj_step(self.model, self.data)
            
            # Generate frame every 2nd iteration and when clients are connected
            frame_counter += 1
            if frame_counter % 2 == 0 and self.clients:
                frame_data = self.get_frame_as_base64()
                if frame_data:
                    with self.frame_lock:
                        self.latest_frame = {
                            'type': 'frame',
                            'data': frame_data,
                            'mocap_index': self.current_mocap_index
                        }
                    # Debug print
                    if frame_counter % 60 == 0:  # Print every ~1 second
                        print(f"Generated frame for {len(self.clients)} clients")
            
            # Report performance every 10 seconds
            if time.time() - last_perf_report >= 10.0:
                stats = self.perf_monitor.get_stats()
                print(f"Performance: {stats['avg_render_time']:.1f}ms render, {stats['fps']:.1f} FPS, {len(self.clients)} clients")
                last_perf_report = time.time()
            
            rate.sleep()
    
    def handle_control_command(self, command, direction):
        """Handle movement commands"""
        mocap_id = self.mocap_ids[self.current_mocap_index]
        
        if command == 'move_x':
            self.data.mocap_pos[mocap_id][0] += direction * self.step_size
            print(f"Move X: {direction * self.step_size}")
        elif command == 'move_y':
            self.data.mocap_pos[mocap_id][1] += direction * self.step_size
            print(f"Move Y: {direction * self.step_size}")
        elif command == 'move_z':
            self.data.mocap_pos[mocap_id][2] += direction * self.step_size
            print(f"Move Z: {direction * self.step_size}")
        elif command == 'rotate_z':
            # FIX: Correct quaternion multiplication
            current_quat = self.data.mocap_quat[mocap_id].copy()
            current_so3 = mink.SO3(current_quat)
            rotation_so3 = mink.SO3.from_z_radians(direction * 0.1)
            new_so3 = current_so3 @ rotation_so3  # Use @ operator for SO3 multiplication
            self.data.mocap_quat[mocap_id] = new_so3.wxyz
            print(f"Rotate Z: {direction * 0.1}")
    
    async def handle_websocket(self, websocket):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial frame if available
        with self.frame_lock:
            if self.latest_frame:
                try:
                    await websocket.send(json.dumps(self.latest_frame))
                except:
                    pass
        
        # Start frame sender for this client
        frame_task = asyncio.create_task(self.send_frames_to_client(websocket))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                print(f"Received: {data}")  # Debug print
                
                if data['type'] == 'control':
                    self.command_queue.put({
                        'command': data['command'],
                        'direction': data['direction']
                    })
                elif data['type'] == 'switch_mocap':
                    self.current_mocap_index = (self.current_mocap_index + 1) % len(self.mocap_ids)
                    print(f"Switched to mocap index: {self.current_mocap_index}")
                elif data['type'] == 'set_step_size':
                    self.step_size = max(0.001, min(0.1, float(data['value'])))
                    print(f"Step size set to: {self.step_size}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            frame_task.cancel()
            self.clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_frames_to_client(self, websocket):
        """Send frames to a specific client"""
        try:
            while True:
                with self.frame_lock:
                    frame_to_send = self.latest_frame
                
                if frame_to_send:
                    try:
                        await websocket.send(json.dumps(frame_to_send))
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"Error sending frame: {e}")
                
                await asyncio.sleep(0.033)  # ~30 FPS
        except asyncio.CancelledError:
            pass
    
    def start_server(self, host='localhost', port=8765):
        """Start the complete server"""
        # Start simulation thread
        sim_thread = threading.Thread(target=self.simulation_loop)
        sim_thread.daemon = True
        sim_thread.start()
        print("Simulation started")
        
        # Start WebSocket server
        async def main():
            async with websockets.serve(self.handle_websocket, host, port):
                print(f"WebSocket server starting on ws://{host}:{port}")
                await asyncio.Future()  # Run forever
        
        asyncio.run(main())

if __name__ == "__main__":
    server = WebTiagoServer()
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.running = False