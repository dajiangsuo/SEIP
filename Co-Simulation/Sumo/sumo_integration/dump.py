#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the carla simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import logging
from datetime import datetime


import carla  # pylint: disable=import-error

#import h5py
import numpy as np
from queue import Queue, Empty
import os
import open3d as o3d
import yaml

from .perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show

from .perception.obstacle_vehicle import ObstacleVehicle


from .constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z

try:
    import queue
except ImportError:
    import Queue as queue

points_per_cloud = 50000
#fps = 10.0
total_frames = 50000
num_infra_sensor = 1
BURN = 800

#sensor settings
CAMERA_HEIGHT_POS = 5.5 # 18 feet/ 5.4864 meters
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS

# ==================================================================================================
# -- carla simulation ------------------------------------------------------------------------------
# ==================================================================================================


class CarlaSimulation(object):
    """
    CarlaSimulation is responsible for the management of the carla simulation.
    """
    def __init__(self, host, port, step_length):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.step_length = step_length

        # The following sets contain updated information for the current frame.
        self._active_actors = set()
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # this sensor list is created for convenience and can also be useful when destroying sensors. Added by Dajiang Suo
        self.sensor_list = []
        self.sensor_queues = []
        # for vidualization
        #self.o3d_vis = None
        self.o3d_vis_queue = [] 
        #o3d_visualizer_init(lidar_id)
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        self.savedFrames = -BURN
        self.fps = 1. / step_length

        # setting up folders for storing dumped sensing data
        current_path = os.path.dirname(os.path.realpath(__file__))
         # load current time for data dumping and evaluation
        current_time = datetime.now()
        current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_parent_folder = \
            os.path.join(current_path,
                         '../../../data_dumping',
                         current_time)
                         #str(self.vehicle_id))

        if not os.path.exists(self.save_parent_folder):
            os.makedirs(self.save_parent_folder)

        self.sumo2carla_ids = {}
        self.count = 0

        """
        compression_opts = {'compression':'gzip', 'compression_opts':9}
        
        self.f = h5py.File('test.hdf5', 'w')
        self.f.create_dataset('point_cloud', (total_frames, num_infra_sensor, points_per_cloud, 4), dtype='float16', **compression_opts)
        self.f.create_dataset('lidar_pose', (total_frames, num_infra_sensor, 6), dtype='float32', **compression_opts)
        self.f.create_dataset('vehicle_boundingbox', (total_frames, 10, 8),maxshape=(total_frames,None, 8), dtype='float32', **compression_opts)
        #f.create_dataset('pedestrian_boundingbox', (args.frames, args.npedestrians, 8), dtype='float32', **compression_opts)
        """

        # Set traffic lights.
        self._tls = {}  # {landmark_id: traffic_ligth_actor}

        tmp_map = self.world.get_map()
        for landmark in tmp_map.get_all_landmarks_of_type('1000001'):
            if landmark.id != '':
                traffic_ligth = self.world.get_traffic_light(landmark)
                if traffic_ligth is not None:
                    self._tls[landmark.id] = traffic_ligth
                else:
                    logging.warning('Landmark %s is not linked to any traffic light', landmark.id)

    def get_actor(self, actor_id):
        """
        Accessor for carla actor.
        """
        return self.world.get_actor(actor_id)

    # This is a workaround to fix synchronization issues when other carla clients remove an actor in
    # carla without waiting for tick (e.g., running sumo co-simulation and manual control at the
    # same time)
    def get_actor_light_state(self, actor_id):
        """
        Accessor for carla actor light state.

        If the actor is not alive, returns None.
        """
        try:
            actor = self.get_actor(actor_id)
            return actor.get_light_state()
        except RuntimeError:
            return None

    @property
    def traffic_light_ids(self):
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        if landmark_id not in self._tls:
            return None
        return self._tls[landmark_id].state

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                # We set the traffic light to 'green' because 'off' state sets the traffic light to
                # 'red'.
                actor.set_state(carla.TrafficLightState.Green)

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.

            :param blueprint: blueprint of the actor to be spawned.
            :param transform: transform where the actor will be spawned.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z),
                                    transform.rotation)

        batch = [
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False))
        ]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return INVALID_ACTOR_ID

        return response.actor_id

    def spawn_infra_sensor(self,type,x,y,z):
        # inputs
        # type: string
        #       the type of sensor, now we only support lidar and add cam in the future
        # location: carla.Location
        # spawn an infrastructure sensor with the specified transform
        location = carla.Location(x,y,z)
        sensor_pose = carla.Transform(location) # e.g., location is carla.Location(x=-77.8, y=16, z=0)
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(sensor_pose.location)

        infra_sensor_bp = self.world.get_blueprint_library().find('sensor.camera.rgb') # note becuase we don't have a blueprint to represent infrastructure
                                                                                  # sensors, we just use the blueprint of a rgb camera to serve the role

        infra_sensor = self.world.spawn_actor(
            infra_sensor_bp,
            sensor_pose)
        infra_sensor.set_transform(waypoint.transform)

        if type == 'lidar':
            # build lidar bp and transform relative to the road sensor
            """
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('channels', '40')
            lidar_bp.set_attribute('range', str(MAX_RENDER_DEPTH_IN_METERS*100))    #cm to m
            #lidar_bp.set_attribute('points_per_second', '720000')
            lidar_bp.set_attribute('points_per_second', '50000')
            lidar_bp.set_attribute('rotation_frequency', '10.0')
            lidar_bp.set_attribute('upper_fov', '7')
            lidar_bp.set_attribute('lower_fov', '-16')
            #lidar_bp.set_attribute('sensor_tick', '0.0')
            """
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', '0.35')
            lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
            lidar_bp.set_attribute('points_per_second', str(points_per_cloud*self.fps))
            lidar_bp.set_attribute('rotation_frequency', str(self.fps))
            lidar_bp.set_attribute('channels', '32.0')
            lidar_bp.set_attribute('lower_fov', '-30.0')
            lidar_bp.set_attribute('upper_fov', '10.0')
            lidar_bp.set_attribute('range', '80.0')
            lidar_bp.set_attribute('noise_stddev', '0.02')
            lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=LIDAR_HEIGHT_POS),carla.Rotation(yaw=0,pitch=0))
            # add sensor
            lidar = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=infra_sensor)
            self.sensor_list.append(lidar)
            lidar_id = lidar_name = 'lidar%d' % (len(self.sensor_list)-1)
            o3d_vis = o3d_visualizer_init(lidar_id)
            self.o3d_vis_queue.append(o3d_vis)
            # set up a queue for Lidar for storing generated data
            q = queue.Queue()
            lidar.listen(lambda data : q.put(data)) # note here we just put the original Sensor data in the queue, which include, data.frame
                                                    # and data.transform in addition to data.raw_data
            self.sensor_queues.append(q) 

            #actor_list.append(lidar) # this is for destroying the actor, probably unnecessary
        else:
            # build rgb cam bp and transform relative to the road sensor
            cam_rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Modify the attributes of the cam_rgb_bp to set image resolution and field of view.
            cam_rgb_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
            cam_rgb_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
            cam_rgb_bp.set_attribute('fov', '90.0')
            # Set the time in seconds between sensor captures
            #print cam_rgb_bp.get_attribute('sensor_tick')
            #print "*)()()()()()()()()()()()"
            #cam_rgb_bp.set_attribute('sensor_tick', '1.0')
            # Provide the position of the sensor relative to the vehicle.
            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS),carla.Rotation(yaw=0,pitch=0))
            cam = self.world.spawn_actor(
            cam_rgb_bp,
            rgb_transform,
            attach_to=infra_sensor)
            self.sensor_list.append(cam)

    """ Note, these operations ragarding lidar point clouds will be performed when we retrieve sensor data form the queue
    def lidar_callback(self,data):
        points = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.sensorQueue.put((data.frame, point_cloud, data.transform))
    """

    def update_sumo2carla_ids(self,sumo2carla_ids):
        self.sumo2carla_ids = sumo2carla_ids

    def dump_yaml_file(self,snap,frame,vehicle_list,sensor_list):

        # dumping info about the ground trueth of vehicle objects
        dump_yaml = {}
        vehicle_dict = {}



        for veh in vehicle_list:
            #veh_actor_snap = snap.find(vehicle.id)
            #v_transform = veh_actor_snap.get_transform()
            #v_ext = vehicle.bounding_box.extent

            veh_carla_id = veh.id
            #veh_actor_snap = snap.find(vehicle.id)
            veh_actor_snap = snap.find(veh_carla_id)
            veh_pos = veh_actor_snap.get_transform()
            veh_bbx = veh.bounding_box

            vehicle_dict.update({veh_carla_id: {
                #'bp_id': veh.type_id,
                #'color': veh.color,
                "location": [veh_pos.location.x,
                             veh_pos.location.y,
                             veh_pos.location.z],
                "center": [veh_bbx.location.x,
                           veh_bbx.location.y,
                           veh_bbx.location.z],
                "angle": [veh_pos.rotation.roll,
                          veh_pos.rotation.yaw,
                          veh_pos.rotation.pitch],
                "extent": [veh_bbx.extent.x,
                           veh_bbx.extent.y,
                           veh_bbx.extent.z]
                #"speed": veh_speed
            }})

        dump_yaml.update({'vehicles': vehicle_dict})

        # lidar pose under world coordinate system
        for i,lidar in enumerate(sensor_list):
            #extract transform info for each sensor
            lidar_transformation = lidar.get_transform()
            lidar_pose_name = 'lidar%d' % i + '_pose'
            #dump_yaml.update({'lidar_pose': [
            dump_yaml.update({lidar_pose_name: [
                lidar_transformation.location.x,
                lidar_transformation.location.y,
                lidar_transformation.location.z,
                lidar_transformation.rotation.roll,
                lidar_transformation.rotation.yaw,
                lidar_transformation.rotation.pitch]})

        dump_yaml.update({'RSU': True})
        yml_name = '%06d' % frame + '.yaml'

        save_path = os.path.join(self.save_parent_folder,
                                 yml_name)

        #save_yaml(dump_yml, save_path)
        with open(save_path, 'w') as outfile:
            yaml.dump(dump_yaml, outfile, default_flow_style=False)

    def dump_lidar_data(self,snap,pcl,pcl_name):
        #points = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
        #pcl = np.reshape(points, (int(points.shape[0] / 4), 4))

        #. Is this necessary to 
        if points_per_cloud > pcl.shape[0]:
            pcl_pad = np.pad(pcl, ((0, points_per_cloud-pcl.shape[0]),(0,0)), mode='constant')

        point_xyz = pcl_pad[:, :-1]
        point_intensity = pcl_pad[:, -1]
        point_intensity = np.c_[
            point_intensity,
            np.zeros_like(point_intensity),
            np.zeros_like(point_intensity)
        ]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

        o3d.io.write_point_cloud(os.path.join(self.save_parent_folder,
                                              pcl_name),
                                 pointcloud=o3d_pcd,
                                 write_ascii=True)




    def destroy_infra_sensors(self):
        """
        Destroys infrastructure sensors in the sensor_list
        """
        for sensor in self.sensor_list:
            sensor.destroy()
        self.sensor_list.clear()

        return

    def destroy_visualizer(self):
        for i in range(len(self.o3d_vis_queue)):
            #3d_vis.destroy_window()
            self.o3d_vis_queue[i].destroy_window()
        self.o3d_vis_queue.clear()

        return

    def destroy_actor(self, actor_id):
        """
        Destroys the given actor.
        """
        actor = self.world.get_actor(actor_id)
        if actor is not None:
            return actor.destroy()
        return False

    def synchronize_vehicle(self, vehicle_id, transform, lights=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param lights: new vehicle light state.
            :return: True if successfully updated. Otherwise, False.
        """
        vehicle = self.world.get_actor(vehicle_id)
        if vehicle is None:
            return False

        vehicle.set_transform(transform)
        if lights is not None:
            vehicle.set_light_state(carla.VehicleLightState(lights))
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param landmark_id: id of the landmark to be updated.
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        if not landmark_id in self._tls:
            logging.warning('Landmark %s not found in carla', landmark_id)
            return False

        traffic_light = self._tls[landmark_id]
        traffic_light.set_state(state)
        return True

    def tick(self):
        """
        Tick to carla simulation.
        """
        self.world.tick()

        #Create HDF5 file with datasets
        
        # get sensor data from the sensor list
        snap = self.world.get_snapshot()
        if self.savedFrames < total_frames  and self.savedFrames >= 0:

            
            # dump sensor data to files
            for i,current_queue in enumerate(self.sensor_queues): # this is a bad design to seperate sensor queues from sensors
                sensor_data = current_queue.get(True,2.0)
                assert (sensor_data.frame == snap.frame-BURN), f"sensor_data frame:{sensor_data.frame},but current frame:{snap.frame}"
                # dump the lidar data to .pcd files. Note the corresponding lidar transform will be
                # stored in the yaml file
                pcl_name = '%06d' % sensor_data.frame + '_' + 'lidar%d' % i + '.pcd'
                points = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
                pcl = np.reshape(points, (int(points.shape[0] / 4), 4))

                # filter out vehicles outside the Lidar's maximum range
                center_location = carla.Location(x=-50.0, y=0.5, z=0) # the center of the intersection
                vehicle_list = [vehicle for vehicle in self.world.get_actors().filter('vehicle.*') if vehicle.get_location().distance(center_location) < 50]

                # filter out vehicles that actually detected by the LiDAR

                
                """
                # visualize LiDAR data
                obs_vehicle_list = [
                    ObstacleVehicle(
                        None,
                        None,
                        v,
                        self.sensor_list[i],
                        self.sumo2carla_ids) for v in vehicle_list]

                # visualize sensor data
                objects = {'vehicles': []} # all objects we want to visualize. Only consider cars for now
                objects.update({'vehicles': obs_vehicle_list})

                o3d_pointcloud_encode(pcl, self.o3d_pointcloud)
                #o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
                # render the raw lidar
                o3d_visualizer_show(
                    self.o3d_vis_queue[i],
                    self.savedFrames,
                    self.o3d_pointcloud,
                    objects)
                """

                # dump LiDAR data; dump ground truth vehicle position and sensor transform
                if self.savedFrames %10 == 0:
                    self.dump_lidar_data(snap,pcl,pcl_name)

            # dump ground truth (e.g., vehicle position and sensor transform)
            if self.savedFrames %10 == 0:
                self.dump_yaml_file(snap,snap.frame-BURN,vehicle_list,self.sensor_list)
                logging.info(f'World frame {snap.frame} saved succesfully as frame {self.savedFrames}')
                
                    

                

                #write data to file
                #self.f['point_cloud'][self.savedFrames,i] = pcl_pad
                #self.f['lidar_pose'][self.savedFrames, i] = np.array([transform.location.x,transform.location.y,transform.location.z, transform.rotation.pitch,transform.rotation.yaw,transform.rotation.roll])
                #f['vehicle_boundingbox'][savedFrames, i] = np.array([v_transform.location.x,v_transform.location.y,v_transform.location.z+v_ext.z,v_transform.rotation.yaw,v_transform.rotation.pitch,2*v_ext.x,2*v_ext.y,2*v_ext.z])


            """
            for i,vehicle in enumerate(self.world.get_actors().filter('vehicle.*')): #Get the actor and the snapshot information
                # only consider vehicles within the range of 150m from the center of the intersection, otherwise it would be a unfair 
                # comparision with other staff
                veh_actor_snap = snap.find(vehicle.id)
                v_transform = veh_actor_snap.get_transform()
                v_ext = vehicle.bounding_box.extent
                self.f['vehicle_boundingbox'][self.savedFrames, i] = np.array([v_transform.location.x,v_transform.location.y,v_transform.location.z+v_ext.z,v_transform.rotation.yaw,v_transform.rotation.pitch,2*v_ext.x,2*v_ext.y,2*v_ext.z])
            """
        if self.savedFrames < 0:
            logging.info(f'World frame {snap.frame} burnt, {-self.savedFrames} to start recording')
            
            

        self.savedFrames += 1



        # Update data structures for the current frame.
        current_actors = set(
            [vehicle.id for vehicle in self.world.get_actors().filter('vehicle.*')])
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

    def close(self):
        """
        Closes carla client.
        """
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(False)
