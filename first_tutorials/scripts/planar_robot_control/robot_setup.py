import pybullet as p
import pybullet_data
import time
import os

# setup und connect to PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# load urdf with fixed base
# Robot origin pos
ROBOT_ORIGIN = [0,0,0]
ROBOT_URDF="planar_robot.urdf"

startOrientation = p.getQuaternionFromEuler([0, 0, 0])
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, ROBOT_URDF)  # URDF relative to script

plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(urdf_path, ROBOT_ORIGIN, useFixedBase=True) # at origin with a fixed base

# get joint info
num_joints = p.getNumJoints(robot)
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    print(f"Joint {i}: {info[1].decode('utf-8')}")  # prints joint names

# set target positions for joints
target_positions = [0.7, 1.3]  # radians for joint 0 and joint 1

# endless simulation loop
try:
    while True:
        # Apply position control for each joint
        for joint_index, target_pos in enumerate(target_positions):
            p.setJointMotorControl2(
                bodyUniqueId=robot,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500  # max force at the joint (torque)
            )
        
        p.stepSimulation()
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
    # print final joint states
    for i in range(num_joints):
        joint_state = p.getJointState(robot, i)
        print(f"Joint {i} final position: {joint_state[0]}")
    p.disconnect()