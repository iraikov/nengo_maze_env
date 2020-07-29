import nengo
import numpy as np
import scipy.ndimage
from skimage.data import immunohistochemistry, binary_blobs
from skimage.draw import circle
from functools import partial
from mazelab.generators import random_shape_maze, morris_water_maze
from aniso import anisodiff
from enum import Enum
from collections import defaultdict

# need to install mazelab to use this maze generator
# https://github.com/zuoxingdong/mazelab

class MazeShape(Enum):
    MAZE_RANDOM = 1
    MAZE_MORRIS = 2
    MAZE_HANLON = 3

        
def generate_sensor_readings(map_arr,
                             object_map,
                             zoom_level=4,
                             n_sensors=1,
                             fov_rad=np.pi,
                             x=0,
                             y=0,
                             th=0,
                             max_sensor_dist=10,
                            ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    arr_zoom = scipy.ndimage.zoom(map_arr, zoom_level, order=0)

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    readings = []
    for i, ang in enumerate(angs):
        dist, ix, iy = get_collision_coord(arr_zoom, x*zoom_level, y*zoom_level, ang, max_sensor_dist*zoom_level)
        texture = 0.
        if (ix > -1) and (iy > -1):
            sx = ix/zoom_level
            sy = iy/zoom_level
            if int(sx) in object_map:
                if int(sy) in object_map[int(sx)]:
                    texture = object_map[int(sx)][int(sy)]
        readings.append((dist / zoom_level, texture))
    return readings

def get_collision_coord(map_array, x, y, th, max_sensor_dist=10*4,):
    """
    Find the first occupied space given a start point and direction
    Meant for a zoomed in map_array
    """
    # Define the step sizes
    dx = np.cos(th)
    dy = np.sin(th)

    # Initialize to starting point
    cx = x
    cy = y

    for i in range(max_sensor_dist):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy
        cx = np.clip(cx, 0, map_array.shape[0] - 1)
        cy = np.clip(cy, 0, map_array.shape[1] - 1)
        ix = int(cx)
        iy = int(cy)
        if map_array[ix, iy] > 0.:
            return i, ix, iy
        
    return max_sensor_dist, -1, -1

class NengoMazeEnvironment(object):
    """
    Defines a maze environment to be used as a Nengo node
    Takes as input the agent x,y,th state as well as a map generation seed
    """

    def __init__(self,
                 n_sensors,
                 fov=180,
                 height=15,
                 width=15,
                 max_sensor_dist=10,
                 kappa=25,
                 reward_location=(8,8),
                 normalize_sensor_output=False,
                 input_type= 'directional_velocity',
                 dt=0.1,
                 maze_shape=MazeShape.MAZE_RANDOM,
                 maze_kwargs={}
    ):

        # Sets how inputs are interpreted
        # Forced to stay within the bounds
        assert(input_type) in ['position', 'holonomic_velocity', 'directional_velocity']
        self.input_type = input_type

        # dt value for the velocity inputs
        self.dt = dt

        # Number of distance sensors
        self.n_sensors = n_sensors

        self.max_sensor_dist = max_sensor_dist

        # If true, divide distances by max_sensor_dist
        self.normalize_sensor_output = normalize_sensor_output

        # Size of the map
        self.height = height
        self.width = width

        self.x = int(width/2.)
        self.y = int(height/2.)
        self.th = 0

        self.sensor_dists = np.zeros((self.n_sensors,))

        self.fov = fov
        self.fov_rad = fov * np.pi / 180.

        # Save the last seed used so as to not regenerate a new map until needed
        self.current_seed = 0

        self.maze_kwargs = maze_kwargs
        self.maze_shape = maze_shape
        
        # Create the default reward map
        self.reward_location = reward_location
        self._generate_reward_map()

        # Create the default starting texture
        self.kappa = kappa
        self._generate_texture_map()

        # Create the default starting map
        self._generate_map()

        # Set up svg element templates to be filled in later
        self.platform_template =  '<rect x={0} y={1} width=1 height=1 style="fill:none;stroke:black;stroke-width:5;fill-opacity:0.1;stroke-opacity:0.9"/>'
               
        self.tile_template =  '<rect x={0} y={1} width=1 height=1 style="fill:black;"/>' 
        self.object_template =  '<rect x={0} y={1} width=1 height=1 style="fill:black;fill-opacity:{2};"/>' 
        self.reward_template =  '<rect x={0} y={1} width=1 height=1 style="fill:red;fill-opacity:{2};"/>' 
        self.agent_template = '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:blue" transform="translate({0},{1}) rotate({2})"/>'
        self.sensor_template = '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:rgba(128,128,128,{4});stroke-width:.1"/>'
        self.svg_header = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(self.height, self.width)

        # self.cue_symbol = '<rect x={0} y={1} width=10 height=10 style="fill:red"/>'

        self._generate_svg()

    def _generate_map(self):
        """
        Generate a new map based on the current seed
        """
        # TODO: make sure this seed setting actually works
        np.random.seed(self.current_seed)
        self.object_map = defaultdict(lambda: dict())

        if self.maze_shape == MazeShape.MAZE_RANDOM:
            maze = random_shape_maze(width=self.width, height=self.height,
                                     max_shapes=50, max_size=8, allow_overlap=False, shape=None,
                                     seed=self.current_seed)
            
        elif self.maze_shape == MazeShape.MAZE_MORRIS:
            radius = int(self.width / 2) + 1 if self.width <= self.height else int(self.height / 2) + 1
            maze = morris_water_maze(radius, platform_center=(0, 0), platform_radius=1)
            
        elif self.maze_shape == MazeShape.MAZE_HANLON:
            
            radius = int(self.width / 2) + 1 if self.width <= self.height else int(self.height / 2) + 1
            maze = morris_water_maze(radius, platform_center=(self.width/2, self.height/2), platform_radius=0.)
            n_objects = self.maze_kwargs.get("n_objects", 4)
            object_radius = self.maze_kwargs.get("object_radius", 1.)
            for i in range(n_objects):
                while True:
                    x, y = np.random.randint(1, self.width), np.random.randint(1, self.height)
                    if (maze[x, y] == 0) and not ((x,y) == self.reward_location):
                        maze[x, y] = self.texture_map[x, y]
                        self.object_map[x][y] = self.texture_map[x, y]
                        break
            #print(self.object_map)

        else:
            raise RuntimeError("NengoMazeEnvironment: unknown maze shape %s" % str(self.maze_shape))

        self.maze = maze

    def _generate_texture_map(self, zoom_level=4):
        """
        Generate a new texture map based on the current seed
        """
        np.random.seed(self.current_seed+1)
        
        texture_map = binary_blobs(length=max(self.width, self.height),n_dim=2)

        result = anisodiff(texture_map, niter=10, option=1, kappa=self.kappa)

        self.texture_map = result

    def _generate_reward_map(self):
        """
        Generate a new reward map based on the current seed
        """
        x, y = self.reward_location

        self.reward = np.zeros((1,))
        self.reward_map = {(x, y): 1.0}


        
    def _generate_svg(self):

        # TODO: make sure coordinates are correct (e.g. inverted y axis)
        # NOTE: x and y currently flipped from: https://github.com/tcstewar/ccmsuite/blob/master/ccm/ui/nengo.py
        # draw tiles
        tiles = []
        for i in range(self.height):
            for j in range(self.width):
                # For simplicity and efficiency, only draw the walls and not the empty space
                # This will have to change when tiles can have different colours
                if (i in self.object_map) and (j in self.object_map[i]):
                    tiles.append(self.object_template.format(i, j, abs(self.object_map[i][j])))
                elif (i,j) in self.reward_map:
                    tiles.append(self.reward_template.format(i, j, self.reward_map[(i,j)]))
#                elif self.maze[i, j] == 3.:
#                    tiles.append(self.platform_template.format(i, j))
                elif self.maze[i, j] == 1.:
                    tiles.append(self.tile_template.format(i, j))

                    
        # draw agent
        direction = self.th * 180. / np.pi + 90. #TODO: make sure angle conversion is correct
        x = self.x
        y = self.y
        th = self.th
        agent = self.agent_template.format(x, y, direction)
        # symbol = self.cue_symbol.format(self.height, self.width, direction)

        svg = self.svg_header

        svg += ''.join(tiles)

        # draw distance sensors
        lines = []
        sensor_readings = generate_sensor_readings(
            map_arr=self.maze,
            object_map=self.object_map,
            zoom_level=8,
            n_sensors=self.n_sensors,
            fov_rad=self.fov_rad,
            x=x,
            y=y,
            th=th,
            max_sensor_dist=self.max_sensor_dist,
        )
        self.sensor_dists = np.asarray([ x[0] for x in sensor_readings ])
        self.sensor_textures = np.asarray([ x[1] for x in sensor_readings ])
        
        ang_interval = self.fov_rad / self.n_sensors
        start_ang = -self.fov_rad/2. + th

        for i, dist in enumerate(self.sensor_dists):
            sx = dist*np.cos(start_ang + i*ang_interval) + self.x
            sy = dist*np.sin(start_ang + i*ang_interval) + self.y
            texture = self.sensor_textures[i]
            lines.append(self.sensor_template.format(self.x, self.y, sx, sy, 1.-texture))
        svg += ''.join(lines)

        svg += agent
        # svg += symbol
        svg += '</svg>'

        self._nengo_html_ = svg

    def __call__(self, t, v):

        x = self.x
        y = self.y
        th = self.th
        
        if self.input_type == 'holonomic_velocity':
            x += v[0] * self.dt
            y += v[1] * self.dt
        elif self.input_type == 'directional_velocity':
            #NOTE: the second input is unused in this case
            self.th += v[2] * self.dt
            x += np.cos(self.th) * v[0] * self.dt
            y += np.sin(self.th) * v[0] * self.dt
        elif self.input_type == 'position':
            x = v[0]
            y = v[1]
            self.th = v[2]

        dth = (th - self.th) / self.dt
        
        zoom_level = 2
        map_array = scipy.ndimage.zoom(self.maze, zoom_level, order=0)
        cx = int(x * zoom_level)
        cy = int(y * zoom_level)
        dx = 0.
        dy = 0.
        if map_array[cx, cy] == 0.:
            dx = (self.x - x) / self.dt
            dy = (self.y - y) / self.dt
            self.x = x
            self.y = y

        # Keep the agent within the bounds of the maze
        self.x = np.clip(self.x, 1, self.width - 1)
        self.y = np.clip(self.y, 1, self.height - 1)
        dth =
        if self.th > 2*np.pi:
            self.th -= 2*np.pi
        elif self.th < -2*np.pi:
            self.th += 2*np.pi

        seed = int(v[3])

        if seed != self.current_seed:
            self.current_seed = seed
            self._generate_map()
            self._generate_texture_map()
            self._generate_reward_map()

        # Generate SVG image for nengo_gui
        # sensor_dists is also calculated in this function
        self._generate_svg()

        if self.normalize_sensor_output:
            self.sensor_dists /= self.max_sensor_dist
            
        ix = int(self.x)
        iy = int(self.y)
        self.reward[0] = self.reward_map.get((ix, iy), 0.0)

        #print(self.sensor_textures)
        #return self.sensor_dists
        return np.concatenate([[self.x], [self.y], [dx], [dy],
                               [(self.th / (2*np.pi)) + 1.],
                              self.sensor_dists, self.sensor_textures, self.reward])
