"""
visualize communication and cyber attacks in carla
"""

import carla
import random

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position

class Painter(object):

    def __init__(self,attacker_color=carla.Color(255,0,0),victim_color=carla.Color(0,255,0),ghost_color=carla.Color(75,30,180),detection_color=carla.Color(0,0,120),freq=0.5):
        # defining colors for different visualization
        #self.attacker_color = attacker_color
        self.victim_color = victim_color
        self.ghost_color=ghost_color
        self.detection_color = detection_color
        self.tl_color = carla.Color(255,0,0,0)
        self.v2i_color = carla.Color(0,255,0,0) # blue for V2I communication
        self.attacker_color = carla.Color(0,0,255,0)
        self.attack_sig_color = carla.Color(0,0,255,0)


        self.freq= freq
        self.tl_freq = 0 # note: when setting to zero, tl will be displayed permanently
        self.vehicle_box_dims=carla.Vector3D(1,0.5,0.8)
        self.detection_box_dims=carla.Vector3D(0.05,0.05,0.05)
        self.tl_box_dims = carla.Vector3D(3,0.3,0.4)
        self.attacker_dims = carla.Vector3D(0.5,0.5,2)


    def traffic_light_vis(self,synchronization,tl_landmarks):
    	coloring_brush = synchronization.carla.world.debug
    	for tl_landmark in tl_landmarks:
    		#candidate_tl_landmark_set = list(synchronization.carla.traffic_light_ids)
    		#candidate_tl_landmark = candidate_tl_landmark_set[0]
    		#print("the traffic light landmark to visualize is:",candidate_tl_landmark)
    		#tl_actor = synchronization.carla._tls[candidate_tl_landmark]
    		tl_actor = synchronization.carla._tls[tl_landmark]
    		tl_box_loc = tl_actor.get_transform().location
    		tl_box_loc.x += 7.9998828
    		#tl_box_loc.y += 0.10300831
    		tl_box_loc.z += 6.00340332
    		print("rotation in yaw:",(tl_actor.get_transform()).rotation.yaw)
    		#tl_actor_box_list = tl_actor.get_light_boxes()
    		#tl_actor_box = tl_actor_box_list[0]
    		#tl_actor_box_ext = tl_actor_box.extent
    		#tl_box_dims = carla.Vector3D(tl_actor_box_ext.x,tl_actor_box_ext.y,tl_actor_box_ext.z)
    		coloring_brush.draw_box(carla.BoundingBox(tl_box_loc,self.tl_box_dims),tl_actor.get_transform().rotation, 0.05, self.tl_color,self.tl_freq)
    		#coloring_brush.draw_box(carla.BoundingBox(tl_actor_box.location,tl_box_dims),tl_actor_box.rotation, 3, self.tl_color,self.tl_freq)

    def attacker_roadside_vis(self,synchronization,attacker_transform):
        coloring_brush = synchronization.carla.world.debug
        coloring_brush.draw_box(carla.BoundingBox(attacker_transform.location,self.attacker_dims),\
            attacker_transform.rotation, 0.03, self.attacker_color,self.tl_freq)

    def ghost_veh_vis(self,carla_simulation,ghost_location):
        coloring_brush = carla_simulation.world.debug
        ghost_vhe_transform = carla.Transform(location=ghost_location)
        ghost_vhe_transform.location.x = ghost_vhe_transform.location.x - 15 - 3*random.randint(0, 1)
        ghost_vhe_transform.location.y = ghost_vhe_transform.location.y - 5 - 10*random.uniform(0, 5)
        ghost_vhe_transform.rotation.yaw = 90
        coloring_brush.draw_box(carla.BoundingBox(ghost_vhe_transform.location,\
            self.vehicle_box_dims),ghost_vhe_transform.rotation, 0.5, self.attacker_color,life_time=3)
        

    def v2i_communication_vis(self,carla_simulation,sender,receiver):

	#functions that visualize V2I messages exchange between vehicles and traffic lights
	#inputs:
	#	sender: carla actor
	#	receiver: landmark of TL in carla
	
        coloring_brush = carla_simulation.world.debug
        sender_loc = sender.get_transform().location
        tl_actor = carla_simulation._tls[receiver]
        tl_box_loc = tl_actor.get_transform().location
        tl_box_loc.x += 7.9998828
        #tl_box_loc.y += 0.10300831
        tl_box_loc.z += 6.00340332
        coloring_brush.draw_arrow(
        sender_loc,tl_box_loc,thickness=0.1,life_time=0.5,color=self.v2i_color)
        #tl_box_loc,thickness=0.2,life_time=self.freq,color=self.v2i_color)

    def attack2tl_comm_vis(self,carla_simulation,attacker_loc,receiver):

    #functions that visualize V2I messages exchange between vehicles and traffic lights
    #inputs:
    #   sender: carla actor, a walker
    #   receiver: landmark of TL in carla
    
        coloring_brush = carla_simulation.world.debug
        sender_loc = carla.Location(attacker_loc.x,attacker_loc.y,attacker_loc.z)
        tl_actor = carla_simulation._tls[receiver]
        tl_box_loc = tl_actor.get_transform().location
        tl_box_loc.x += 7.9998828
        #tl_box_loc.y += 0.10300831
        tl_box_loc.z += 6.00340332
        sender_loc.z += 1
        coloring_brush.draw_arrow(
        sender_loc,tl_box_loc,thickness=0.1,life_time=0.5,color=self.attack_sig_color)
        #tl_box_loc,thickness=0.2,life_time=self.freq,color=self.v2i_color)

