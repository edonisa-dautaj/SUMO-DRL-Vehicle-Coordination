import math
import traci

def calculate_distance(vehicle_1, vehicle_2):
    x1, y1 = traci.vehicle.getPosition(vehicle_1)
    x2, y2 = traci.vehicle.getPosition(vehicle_2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class Collisions_veh:
    def __init__(self):
        self.collided_vehicles = []

    def detect_collisions(self):
        collisions = traci.simulation.getCollidingVehiclesIDList()
        if collisions :
            self.collided_vehicles.append(collisions[0])
            self.collided_vehicles.append(collisions[1])
            print(f"Collisions détectées : {collisions}")
        return list(set(self.collided_vehicles))

    def detect_collisions_in_junction(self,min_distance):
        junction_id = ['clusterJ4_J6_J7', ':clusterJ4_J6_J7_4_0', 'J0', 'J1', 'J5', ':clusterJ4_J6_J7_0',':clusterJ4_J6_J7_1',':clusterJ4_J6_J7_2',':clusterJ4_J6_J7_3',':clusterJ4_J6_J7_4', 'clusterJ3_J6']
        vehicles = traci.vehicle.getIDList()
        vehicles_in_junction = [
            vehicle for vehicle in vehicles if traci.vehicle.getRoadID(vehicle) in junction_id
        ]
        for i, vehicle_1 in enumerate(vehicles_in_junction):
            for vehicle_2 in vehicles_in_junction[i+1:]:
                distance = calculate_distance(vehicle_1, vehicle_2)
                if distance < min_distance:
                    self.collided_vehicles.append(vehicle_1)
                    self.collided_vehicles.append(vehicle_2)
                    print(f"Collisions détectées dans la jonction entre :", vehicle_1 , vehicle_2)
        return list(set(self.collided_vehicles))