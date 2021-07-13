from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os

def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml')) #robot configuration file path
#    settings = MeshRendererSettings() #generating renderer settings object
    
    #generating simulator object
    #s = Simulator(mode='headless', #simulating without gui 
    s = Simulator(mode='headless', #simulating with gui
#                  image_width=256, #robot camera pixel width
#                  image_height=256, #robot camera pixel height
#                  vertical_fov=40, #robot camera view angle (from floor to ceiling 40 degrees)
                  )
                  #rendering_settings=settings)
    
    #generating scene object
    scene = InteractiveIndoorScene('Benevolence_1_int', #scene name: Benevolence, floor number: 1, does it include interactive objects: yes (int). I pick Benevolence on purpose as it is the most memory friendly scene in the iGibson dataset. 
                              build_graph=True, #builds the connectivity graph over the given facedown traversibility map (floor plan)
                              waypoint_resolution=0.1, #radial distance between 2 consecutive waypoints (10 cm)
                              pybullet_load_texture=True) #do you want to include texture and material properties? (you need this for object interaction)

    s.import_ig_scene(scene) #loading the scene object in the simulator object
    turtlebot = Turtlebot(config) #generating the robot object
    s.import_robot(turtlebot) #loading the robot object in the simulator object
    init_pos = turtlebot.get_position() #getting the initial position of the robot base [X:meters, Y:meters, Z:meters] (base: robot's main body. it may have links and the associated joints too. links and joints have positions and orientations as well.)
    init_or = turtlebot.get_rpy() #getting the initial Euler angles of the robot base [Roll: radians, Pitch: radians, Yaw: radians]

    #sampling random goal states in a desired room of the apartment
    np.random.seed(0)
    goal = scene.get_random_point_by_room_type('living_room')[1] #sampling a random point in the living room
        
    rnd_path = scene.get_shortest_path(0,init_pos[0:2],goal[0:2],entire_path=True)[0] #generate the "entire" a* path between the initial and goal nodes

    for i in range(len(rnd_path[0])-1):
        with Profiler('Simulator step'): #iGibson simulation loop requieres this context manager
            
            rgb_camera = np.array(s.renderer.render_robot_cameras(modes='rgb')) #probing RGB data, you can also probe rgbd or even optical flow if robot has that property in its config file (.yaml file)
            plt.imshow(rgb_camera[0,:,:,3]) #calling sampled RGB data
            lidar = s.renderer.get_lidar_all() #probing 360 degrees lidar data

            delta_pos = smt_path[:,i+1] - smt_path[:,i] #direction vector between 2 consucutive waypoints
            delta_yaw = np.arctan2(delta_pos[1],delta_pos[0]) #the yaw angle of the robot base while following the sampled bezier path
            delta_qua = e2q(init_or[0],init_or[1],delta_yaw) #transforming robot base Euler angles to quaternion
            turtlebot.set_position_orientation([smt_path[0,i],smt_path[1,i],init_pos[2]], delta_qua) #setting the robot base position and the orientation
            s.step() #proceed one step ahead in simulation time

    s.disconnect()

def e2q(roll, pitch, yaw): #Euler angles to quaternion transformation

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def q2e(x, y, z, w): #quaternion to Euler angles transformation

    c23 = 2*(y*z+w*x)
    c33 = (w*w-x*x-y*y+z*z)
    roll = np.arctan2(c23, c33)

    c13 =2*(x*z-w*y) 
    pitch = (np.arcsin(c13))*(-1)

    c12 = 2*(x*y+w*z)
    c11 = (w*w+x*x-y*y-z*z)
    yaw = np.arctan2(c12, c11)

    return roll, pitch, yaw

if __name__ == '__main__':
    main()
