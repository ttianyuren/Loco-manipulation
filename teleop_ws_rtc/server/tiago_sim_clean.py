from pathlib import Path
import mujoco
import numpy as np
import mink
from mink.contrib import TeleopMocap
from loop_rate_limiters import RateLimiter
import threading
import time

class TiagoSim:
    def __init__(self):
        _HERE = Path(__file__).parent.parent.parent / "sim"
        _XML = _HERE / "models" / "pal_tiago_dual" / "tiago_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(_XML))
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.base_mid = self.model.body("base_target").mocapid[0]
        self.l_mid = self.model.body("left_gripper_target").mocapid[0]
        self.r_mid = self.model.body("right_gripper_target").mocapid[0]
        self.key_callback = TeleopMocap(self.data)
        self.running = True
        self.last_frame = None
        self.lock = threading.Lock()
        self.frequency = 200.0  # Video frame rate
        self.step_size = 0.01  # Default step size for mocap movement
        self.current_mocap_index = 0  # Index for cycling through mocap targets
        self.mocap_ids = [self.base_mid, self.l_mid, self.r_mid]  # List of mocap IDs for targets
        # Latency monitoring variables
        self.command_timestamps = {}  # Track command processing times
        self.frame_count = 0
        self.last_frame_time = time.time()
        self._setup_sim()
        self.sim_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.sim_thread.start()

    def _setup_sim(self):
        _JOINT_NAMES = [
            "1_joint", "2_joint", "3_joint", "4_joint", "5_joint", "6_joint", "7_joint"
        ]
        _VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}
        joint_names = []
        velocity_limits = {}
        for prefix in ["left", "right"]:
            for n in _JOINT_NAMES:
                name = f"arm_{prefix}_{n}"
                joint_names.append(name)
                velocity_limits[name] = _VELOCITY_LIMITS[n]
        joint_names.extend(["base_x","base_y","base_th","torso_lift_joint"])
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(f"{name}_position").id for name in joint_names])
        # Initialize task attributes individually for clarity
        self.base_task = mink.FrameTask(
            frame_name="base_link", frame_type="body", position_cost=1.0, orientation_cost=1.0)
        self.l_ee_task = mink.FrameTask(
            frame_name="left_gripper", frame_type="site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.r_ee_task = mink.FrameTask(
            frame_name="right_gripper", frame_type="site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
        self.posture_task = mink.PostureTask(self.model, cost=1e-1)
        self.tasks = [self.base_task, self.l_ee_task, self.r_ee_task, self.posture_task]
        self.lift_subtree_id = self.model.body("torso_lift_link").id
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, velocity_limits),
        ]
        self.solver = "daqp"
        self.pos_threshold = 5e-3
        self.ori_threshold = 5e-3
        self.max_iters = 5
        # Correct MuJoCo API calls
        self.model.key("neutral_pose")  # Ensure key exists
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("neutral_pose").id)
        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(self.model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.base_task.set_target_from_configuration(self.configuration)
        mink.move_mocap_to_frame(self.model, self.data, "left_gripper_target", "left_gripper", "site")
        mink.move_mocap_to_frame(self.model, self.data, "right_gripper_target", "right_gripper", "site")

    def _run_loop(self):
        rate = RateLimiter(frequency=200.0, warn=False)
        while self.running:
            base_pose = self.data.mocap_pos[self.base_mid].copy()
            base_pose[2] = 0
            self.data.mocap_pos[self.base_mid] = base_pose

            # Always set all three task targets from mocap positions
            self.base_task.set_target(mink.SE3.from_mocap_id(self.data, self.base_mid))
            self.l_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.l_mid))
            self.r_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.r_mid))
            # Set targets for all tasks: active from mocap, others from configuration
            if self.current_mocap_index == 0:
                self.base_task.set_target(mink.SE3.from_mocap_id(self.data, self.base_mid))
                self.l_ee_task.set_target_from_configuration(self.configuration)
                self.r_ee_task.set_target_from_configuration(self.configuration)
            elif self.current_mocap_index == 1:
                self.l_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.l_mid))
                self.base_task.set_target_from_configuration(self.configuration)
                self.r_ee_task.set_target_from_configuration(self.configuration)
            elif self.current_mocap_index == 2:
                self.r_ee_task.set_target(mink.SE3.from_mocap_id(self.data, self.r_mid))
                self.base_task.set_target_from_configuration(self.configuration)
                self.l_ee_task.set_target_from_configuration(self.configuration)
            self.key_callback.auto_key_move()
            for i in range(self.max_iters):
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    rate.dt,
                    self.solver,
                    limits=self.limits,
                    damping=1e-5,
                )
                self.configuration.integrate_inplace(vel, rate.dt)
                l_err = self.l_ee_task.compute_error(self.configuration)
                l_pos_achieved = np.linalg.norm(l_err[:3]) <= self.pos_threshold
                l_ori_achieved = np.linalg.norm(l_err[3:]) <= self.ori_threshold
                r_err = self.r_ee_task.compute_error(self.configuration)
                r_pos_achieved = np.linalg.norm(r_err[:3]) <= self.pos_threshold
                r_ori_achieved = np.linalg.norm(r_err[3:]) <= self.ori_threshold
                if (
                    l_pos_achieved and l_ori_achieved and r_pos_achieved and r_ori_achieved
                ):
                    break
            self.data.ctrl[self.actuator_ids] = self.configuration.q[self.dof_ids]
            self._compensate_gravity([self.lift_subtree_id])
            mujoco.mj_step(self.model, self.data)
            # Render frame for video at lower frequency
            if int(time.time() * self.frequency) % 1 == 0:
                with self.lock:
                    self.last_frame = self._render_frame()
                    self.frame_count += 1
            rate.sleep()

    def _compensate_gravity(self, subtree_ids):
        qfrc_applied = self.data.qfrc_applied
        qfrc_applied[:] = 0.0
        jac = np.empty((3, self.model.nv))
        for subtree_id in subtree_ids:
            total_mass = self.model.body_subtreemass[subtree_id]
            mujoco.mj_jacSubtreeCom(self.model, self.data, jac, subtree_id)
            qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac

    def step(self, control):
        # control: dict with keys like 'command' and optionally 'direction' or 'value'
        cmd = control.get("command")
        direction = control.get("direction", 1)
        value = control.get("value")

        # Map high-level teleop commands to low-level actions
        command_map = {
            "forward":   ("move_x",  1),
            "backward":  ("move_x", -1),
            "left":      ("move_y",  1),
            "right":     ("move_y", -1),
        }
        if cmd in command_map:
            cmd, direction = command_map[cmd]

        if cmd in ("cycle_mocap", "switch_mocap"):
            self.current_mocap_index = (self.current_mocap_index + 1) % len(self.mocap_ids)
        elif cmd == "increase_step":
            self.step_size += 0.01
        elif cmd == "decrease_step":
            self.step_size = max(0.001, self.step_size - 0.01)
        elif cmd == "set_step_size":
            try:
                self.step_size = float(value)
            except Exception:
                pass
        elif cmd in ("move_x", "move_y", "move_z"):
            mocap_id = self.mocap_ids[self.current_mocap_index]
            idx = {"move_x": 0, "move_y": 1, "move_z": 2}[cmd]
            self.data.mocap_pos[mocap_id][idx] += direction * self.step_size
        elif cmd == "rotate_z":
            mocap_id = self.mocap_ids[self.current_mocap_index]
            current_quat = self.data.mocap_quat[mocap_id].copy()
            current_so3 = mink.SO3(current_quat)
            rotation_so3 = mink.SO3.from_z_radians(direction * 0.1)
            new_so3 = current_so3 @ rotation_so3
            self.data.mocap_quat[mocap_id] = new_so3.wxyz

    def get_frame(self):
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
            else:
                # If no frame yet, render one
                return self._render_frame()

    def get_frame_base64(self):
        import io, base64
        from PIL import Image
        rgb = self.get_frame()
        img = Image.fromarray(np.flipud(rgb))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=100)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def _render_frame(self):
        # Use MuJoCo's offscreen renderer to get RGB image
        try:
            # Use smaller resolution to fit within framebuffer limits
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            renderer.update_scene(self.data)
            rgb = renderer.render()
            return rgb
        except Exception as e:
            print(f"Render error: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def shutdown(self):
        self.running = False
        self.sim_thread.join()

    def track_command_latency(self, command_id, client_timestamp, websocket=None):
        """Track command processing latency"""
        if client_timestamp:
            # Get server time in same format as client (milliseconds since epoch)
            server_time = time.time() * 1000
            
            # Calculate processing time (server time - client time)
            # Note: This measures network latency + processing time
            processing_time = server_time - client_timestamp
            
            # Only log reasonable latency values (avoid corrupted timestamps)
            if 0 < processing_time < 10000:  # Between 0ms and 10 seconds
                print(f"Command {command_id} processing time: {processing_time:.1f}ms")
                
                # Send the real command latency back to the client
                if websocket and hasattr(websocket, 'send_str'):
                    import asyncio
                    import json
                    latency_response = {
                        'type': 'command_latency',
                        'latency': processing_time,
                        'command': command_id
                    }
                    # Send asynchronously if possible
                    try:
                        asyncio.create_task(websocket.send_str(json.dumps(latency_response)))
                    except:
                        pass  # Ignore send errors
                        
            else:
                # Timestamp corruption detected - use local measurement instead
                if not hasattr(self, '_command_start_time'):
                    self._command_start_time = server_time
                local_processing_time = server_time - self._command_start_time
                self._command_start_time = server_time
                if local_processing_time < 1000:  # Less than 1 second
                    print(f"Command {command_id} local processing time: {local_processing_time:.1f}ms")
        else:
            # No client timestamp - measure local processing time
            current_time = time.time() * 1000
            if hasattr(self, '_last_command_time'):
                local_latency = current_time - self._last_command_time
                if local_latency < 1000:  # Reasonable value
                    print(f"Command {command_id} local latency: {local_latency:.1f}ms")
            self._last_command_time = current_time
    
    def get_performance_stats(self):
        """Get simulation performance statistics"""
        current_time = time.time()
        fps = self.frame_count / (current_time - self.last_frame_time) if current_time != self.last_frame_time else 0
        return {
            'fps': fps,
            'sim_frequency': self.frequency,
            'active_mocap': self.current_mocap_index
        }

# if __name__ == "__main__":
#     sim = TiagoSim()
#     try:
#         while True:
#             time.sleep(1)  # Keep the main thread alive
#     except KeyboardInterrupt:
#         sim.shutdown()
#         print("Simulation stopped.")
