import pybullet as p

"""
Reference Info:

For axes: x - red, y - green, z - blue
pos, orn = p.getBasePositionAndOrientation(ball_id)     # position (x,y,z), orientation (quaternion)
lin_vel, ang_vel = p.getBaseVelocity(ball_id)           # linear and angular velocities

returns:
pos     [x, y, z]
orn     [x, y, z, w] quaternion
lin_vel [vx, vy, vz]
ang_vel [wx, wy, wz]
"""

def push0(ball_id, step):
    if step > 500:
        pos, orn = p.getBasePositionAndOrientation(ball_id)
        if pos[0] <= 1:
            p.applyExternalForce(
                ball_id,                # bodyUniqueId: the ID of the object you want to apply the force to
                -1,                     # linkIndex: -1 means apply force to the base link (for single-link objects)
                forceObj=[5, 5, 0],     # The force vector (Fx, Fy, Fz) in Newtons
                posObj=[0, 0, 0],       # The point of application of the force relative to the link’s frame
                flags=p.WORLD_FRAME     # Frame in which the force is defined. Can also use p.LINK_FRAME. Keep World frame for now
            )
        elif pos[0] > 1:
            p.applyExternalForce(
            ball_id,
            -1,
            forceObj=[-5, -5, 0],  
            posObj=[0, 0, 0],      
            flags=p.WORLD_FRAME     
            )
