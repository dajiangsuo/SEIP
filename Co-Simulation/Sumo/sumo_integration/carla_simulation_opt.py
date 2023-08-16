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

import os
import logging
import yaml
import json
import weakref
from datetime import datetime
from queue import Queue, Empty
from pathlib import Path

import carla  # pylint: disable=import-error
import numpy as np
import open3d as o3d

from .constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z


#points_per_cloud = 50000
# config for type 0
#points_per_second_16 = 300000
points_per_cloud_16 = 30000

range_16 = 100
lower_fov_16 = -15
upper_fov_16 = 15


# config for type 1
#points_per_second_32 = 1000000
#points_per_cloud_32 =  100000
#points_per_second_32 = 600000
points_per_cloud_32 =  60000

range_32 = 200
lower_fov_32 = -25
upper_fov_32 = 15

# config for type 2
#points_per_second_128 = 4000000
#points_per_cloud_128 = 400000
#points_per_second_128 = 900000
points_per_cloud_128 = 90000

range_128 = 300
lower_fov_128 = -25
upper_fov_128 = 15

#fps = 10.0
#total_frames = 50000
TOTAL_FRAMES = 10000
#num_infra_sensor = 1
BURN = 200

#sensor settings
CAMERA_HEIGHT_POS = 5.4 # 18 feet / 5.4864 meters
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS


# ==================================================================================================
# -- carla simulation ------------------------------------------------------------------------------
# ==================================================================================================


class CarlaSimulation(object):
    """
    CarlaSimulation is responsible for the management of the carla simulation.
    """
    def __init__(self, host, port, step_length, path_name):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.step_length = step_length

        self.start_frame = self.world.get_snapshot().frame
        #self.framecount = self.start_frame

        # The following sets contain updated information for the current frame.
        self._active_actors = set()
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # this sensor list is created for convenience and can also be useful when destroying sensors. Added by Dajiang Suo
        self.sensor_list = []
        self.sensor_queues = []

        self.o3d_vis_queue = []

        self.fps = 1. / step_length


        current_time = datetime.now()
        current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.save_dir = Path(f"../../data_dumping/{path_name}")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        self.sumo2carla_ids = {}

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

        # Change this parameter based on LiDAR choosed
        self.points_per_cloud = points_per_cloud_32

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

    def update_sumo2carla_ids(self, sumo2carla_ids):
        self.sumo2carla_ids = sumo2carla_ids

    def spawn_actor(self, blueprint, transform):
        """
        Spawns a new actor.

            :param blueprint: blueprint of the actor to be spawned.
            :param transform: transform where the actor will be spawned.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        transform = carla.Transform(
            transform.location + carla.Location(0, 0, SPAWN_OFFSET_Z),
            transform.rotation,
        )

        batch = [
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)
            )
        ]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error('Spawn carla actor failed. %s', response.error)
            return INVALID_ACTOR_ID

        return response.actor_id

    def spawn_infra_sensor(self, typ: str, x: int, y: int, z: int):
        # inputs
        # typ: string
        #       the type of sensor, now we only support lidar and add cam in the future
        # location: carla.Location
        # spawn an infrastructure sensor with the specified transform

        location = carla.Location(x, y, z)
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(location)

        if typ == 'lidar':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', '0.35')
            lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
            lidar_bp.set_attribute('points_per_second', str(points_per_cloud_32*self.fps))
            lidar_bp.set_attribute('rotation_frequency', str(self.fps))
            lidar_bp.set_attribute('channels', '128.0')
            lidar_bp.set_attribute('lower_fov', '-25.0')
            lidar_bp.set_attribute('upper_fov', '15.0')
            lidar_bp.set_attribute('range', '300.0')
            lidar_bp.set_attribute('noise_stddev', '0.02')

            lidar_location = carla.Location(
                x=waypoint.transform.location.x,
                y=waypoint.transform.location.y,
                z=waypoint.transform.location.z + LIDAR_HEIGHT_POS,
            )
            lidar_transform = carla.Transform(lidar_location, waypoint.transform.rotation)

            lidar = self.world.spawn_actor(
                lidar_bp, lidar_transform,
            )

            lidar_idx = len(self.sensor_list)
            self.sensor_list.append(lidar)
            self.sensor_queues.append(Queue())

            weak_self = weakref.ref(self)
            lidar.listen(
                lambda sensor_data: self.lidar_sensor_callback(weak_self, lidar_idx, sensor_data)
            )
        else:
            raise NotImplementedError(typ)

    @staticmethod
    def lidar_sensor_callback(weak_self, lidar_idx, sensor_data):
        #return
        self: CarlaSimulation = weak_self()
        if self is None:
            return

        lidar = self.sensor_list[lidar_idx]

        snap = self.world.get_snapshot()
        frame = sensor_data.frame

        if frame - self.start_frame > TOTAL_FRAMES:
            print("sensor data generation finished")
            return


        if frame - self.start_frame > BURN and frame % 5 == 0:
            actors = self.world.get_actors()
            vehicle_dict = {}

            center_location = carla.Location(x=-50.0, y=0.5, z=0)

            for vehicle in actors.filter("vehicle.*"):
                if vehicle.get_location().distance(center_location) > 150:
                    continue

                vehicle_snap = snap.find(vehicle.id)
                pose = vehicle_snap.get_transform()
                bbox = vehicle.bounding_box
                vehicle_dict[vehicle.id] = {
                    "location": [
                        pose.location.x,
                        pose.location.y,
                        pose.location.z,
                    ],
                    "rotation": [
                        pose.rotation.pitch,
                        pose.rotation.yaw,
                        pose.rotation.roll,
                    ],
                    "center": [
                        bbox.location.x,
                        bbox.location.y,
                        bbox.location.z,
                    ],
                    "extent": [
                        bbox.extent.x,
                        bbox.extent.y,
                        bbox.extent.z,
                    ],
                }

            pose = lidar.get_transform()
            lidar_pose = [
                pose.location.x,
                pose.location.y,
                pose.location.z,
                pose.rotation.pitch,
                pose.rotation.yaw,
                pose.rotation.roll,
            ]

            with open(self.save_dir / f"{frame:06d}_lidar{lidar_idx}.yaml", "w") as f:
                yaml.dump(
                    {
                        "lidar_pose": lidar_pose,
                        "vehicles": vehicle_dict,
                    },
                    f,
                    yaml.SafeDumper,
                )

            points = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.float32))
            points = np.reshape(points, (-1, 4))

            if points.shape[0] < self.points_per_cloud:
                points = np.pad(
                    points, ((0, self.points_per_cloud - points.shape[0]), (0, 0)), mode='constant'
                )

            points_xyz = points[:, :3]
            points_intensity = points[:, 3]
            points_intensity = np.c_[
                points_intensity,
                np.zeros_like(points_intensity),
                np.zeros_like(points_intensity),
            ]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_xyz)
            pcd.colors = o3d.utility.Vector3dVector(points_intensity)

            o3d.io.write_point_cloud(
                str(self.save_dir / f"{frame:06d}_lidar{lidar_idx}.pcd"),
                pointcloud=pcd,
            )

        self.sensor_queues[lidar_idx].put(frame)

    def destroy_infra_sensors(self):
        """
        Destroys infrastructure sensors in the sensor_list
        """
        for sensor in self.sensor_list:
            sensor.destroy()
        self.sensor_list.clear()

    def destroy_visualizer(self):
        for i in range(len(self.o3d_vis_queue)):
            self.o3d_vis_queue[i].destroy_window()
        self.o3d_vis_queue.clear()

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

        try:
            for i, sensor_queue in enumerate(self.sensor_queues):
                s_frame = sensor_queue.get(True, 20.0)
                print(f"Frame: {s_frame}    Sensor: lidar{i}")
        except Empty:
            print("Warning: Some of the sensor information is missed")

        # Update data structures for the current frame.
        current_actors = set(
            [vehicle.id for vehicle in self.world.get_actors().filter('vehicle.*')]
        )
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
