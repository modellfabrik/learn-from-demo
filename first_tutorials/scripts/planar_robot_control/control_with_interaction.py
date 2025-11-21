import pybullet as p
import pybullet_data
import time
import math
import os

# robot origin:
ROBOT_ORIGIN = [0,0,0]
# joint indices you want to control (wrt. urdf)
JOINT1 = 0
JOINT2 = 1
# end-effector (EE) link index (Currently broken urdf: EE would usually be at `index = JOINT2 + 1`)
EE = 1

# sim setup and conncet to sim runtime
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
# Load urdfs
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "planar_robot.urdf")  # URDF relative to script
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_path, ROBOT_ORIGIN, useFixedBase=True) # at origin with a fixed base

def add_ball():
    # add physics geometries (sphere/ball in this case)
    ball_radius = 0.025
    ball_mass = 0.1
    ball_x, ball_y = -0.1, -0.2
    ball_start_pos = [ball_x, ball_y, ball_radius]  # x, y, z
    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1])
    ball_id = p.createMultiBody(baseMass=ball_mass,
                                baseCollisionShapeIndex=ball_collision,
                                baseVisualShapeIndex=ball_visual,
                                basePosition=ball_start_pos)
    
def control_robot():
    while True:
        # angular inputs for joint1 and joint2 in degrees
        j1_deg = float(input("joint1 (deg): "))
        j2_deg = float(input("joint2 (deg): "))

        T = 5 # seconds, delay to switch to pybullet window if working on vscode.
        print(f"Waiting {T} seconds...")
        time.sleep(T)

        # convert to radians
        j1 = math.radians(j1_deg)
        j2 = math.radians(j2_deg)

        # apply joint commands using position control
        p.setJointMotorControl2(robot, JOINT1, p.POSITION_CONTROL, targetPosition=j1, force=1000)
        p.setJointMotorControl2(robot, JOINT2, p.POSITION_CONTROL, targetPosition=j2, force=1000)

        max_steps = 200
        print(f"Simulation running for {max_steps} steps.")
        # run the simulation steps
        for _ in range(max_steps):
            p.stepSimulation()
            time.sleep(1/60)

        # calculate the forward kinematics (done through built-in `getLinkState` method from Pybullet)
        state = p.getLinkState(robot, EE, computeForwardKinematics=True)
        ee_pos = state[4]   # world position
        ee_orn = state[5]   # world quaternion

        print("FK End-Effector Position:", ee_pos)
        print("FK End-Effector Orientation (quat):", ee_orn)

if __name__ == "__main__":
    add_ball()
    control_robot()