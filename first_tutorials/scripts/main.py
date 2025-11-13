import pybullet as p
import pybullet_data
import time
import numpy as np
import sys
import os

# Import "push" functions
push_functions = []

try:
    from module0 import push0
    push_functions.append(push0)
except Exception as e:
    print(f"Could not import push0: {e}")

for i in range(1, 6):
    try:
        mod = __import__(f"module{i}")
        push_fn = getattr(mod, f"push{i}")
        push_functions.append(push_fn)
    except Exception as e:
        print(f"Module{i} / push{i} not found or failed to import: {e}")


class BallSimulation:
    def __init__(self):
        p.connect(p.GUI, options="--disable-threading --logtostderr")
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        self.time_step = 1./240.
        p.setTimeStep(self.time_step)

        # Simulation parameters
        self.ball_start_pos = [0, 0, 0.4]
        self.goal_center = [1.8, 1.8, 0.1]
        self.goal_radius = 0.1
        self.ball_radius = 0.05
        self.max_steps = 10000

        # Setup environment
        self._setup_plane()
        self._setup_areas()
        self._setup_ball()
        self._setup_maze()

        # Push functions
        self.push_functions = push_functions

    def _setup_plane(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def _setup_areas(self):
        # Start area: blue cylinder
        start_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.goal_radius,
            length=0.01,
            rgbaColor=[0,0,1,0.7]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=start_visual,
                          basePosition=[self.ball_start_pos[0], self.ball_start_pos[1], 0.005])

        # Goal area: green cylinder
        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.goal_radius,
            length=0.01,
            rgbaColor=[0,1,0,0.7]
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual,
                          basePosition=[self.goal_center[0], self.goal_center[1], 0.005])

    def _setup_ball(self):
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=[1,0,0,1])
        self.ball_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=collision,
                                         baseVisualShapeIndex=visual, basePosition=self.ball_start_pos)
        p.changeDynamics(self.ball_id, -1, restitution=0.9, lateralFriction=0.5)

    def _setup_maze(self):
        self.maze_walls = []
        wall_height = 0.2

        # Each entry: [x, y, z, half_x, half_y, half_z, yaw_angle_in_degrees]
        wall_data = [
            # Outer walls
            [ 2.0,  0.0, wall_height/2, 0.05, 2.0, wall_height/2,   0],   # right wall
            [-2.0,  0.0, wall_height/2, 0.05, 2.0, wall_height/2,   0],   # left wall
            [ 0.0,  2.0, wall_height/2, 2.0, 0.05, wall_height/2,   0],   # top wall
            [ 0.0, -2.0, wall_height/2, 2.0, 0.05, wall_height/2,   0],   # bottom wall

            # Inner maze obstacles (static layout)
            #[ 0.5,  1.0, wall_height/2, 0.4, 0.05, wall_height/2,  30],   # angled barrier
            #[-1.5,  0.8, wall_height/2, 0.6, 0.05, wall_height/2, -45],   # another angled barrier
            #[ 0.75, -0.8, wall_height/2, 0.7, 0.05, wall_height/2,   0],   # horizontal wall
            #[ 1.0,  1.4, wall_height/2, 0.3, 0.05, wall_height/2,  90],   # vertical post
            #[-1.0,  1.2, wall_height/2, 0.3, 0.05, wall_height/2, -60],   # diagonal wall
        ]

        for x, y, z, hx, hy, hz, yaw in wall_data:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0, 0, 1, 1])
            orientation = p.getQuaternionFromEuler([0, 0, np.deg2rad(yaw)])
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            self.maze_walls.append(wall_id)



    def run(self):
        if not self.push_functions:
            print("No push functions available. Simulation will run without any forces.")

        step = 0
        while step < self.max_steps:
            # Call each push function safely
            for push_fn in self.push_functions:
                try:
                    push_fn(self.ball_id, step)
                except Exception as e:
                    print(f"Error in {push_fn.__name__} at step {step}: {e}")

            p.stepSimulation()

            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            dist_to_goal = np.linalg.norm(np.array(ball_pos[:2]) - np.array(self.goal_center[:2]))
            if dist_to_goal <= self.goal_radius:
                break

            step += 1
            time.sleep(self.time_step)
            print(f"Current step: {step} - Goal pos: {self.goal_center} - Ball pos: [{ball_pos[0]:.2f},{ball_pos[1]:.2f},{ball_pos[2]:.2f}] - Distance to goal: {dist_to_goal:.3f}", end='\r')
        else:
            print("\nBall did not reach goal within maximum steps.")

        # Disconnect physics and print final message last
        p.disconnect()
        if dist_to_goal <= self.goal_radius:
            print(f"\nSuccess! Ball reached goal at step {step}, position: {ball_pos}")
        else:
            print("\nSimulation finished. Ball did not reach goal.")


if __name__ == "__main__":
    sim = BallSimulation()
    sim.run()