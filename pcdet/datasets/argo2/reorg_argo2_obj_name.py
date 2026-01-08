import pickle
import numpy as np



# label remapping dictionary
mapping = {
    "Regular_vehicle": "car",
    "Pedestrian": "pedestrian",
    "Mobile_pedestrian_crossing_sign": "pedestrian",
    "Box_truck": "truck",
    "Truck": "truck",
    "Truck_cab": "truck",
    "Large_vehicle": "large_vehicle",
    "Stop_sign": "sign",
    "Sign": "sign",
    "Construction_cone": "traffic_cone",
    "Motorcycle": "motorcycle",
    "Vehicular_trailer": "trailer",
    "Traffic_light_trailer": "trailer",
    "Message_board_trailer": "trailer",
    "Motorcyclist": "rider",
    "Wheeled_rider": "rider",
    "Bicyclist": "rider",
    "Bicycle": "bicycle",
    "Bus": "bus",
    "School_bus": "bus",
    "Articulated_bus": "bus",
    "Dog": "dog",
    'Animal': 'animal',
    "Railed_vehicle": "tram",
    'Construction_barrel': 'construction_barrel',
    "Bollard": "others",
    "Wheeled_device": "others",
    "Stroller": "others",
    "Wheelchair": "others",
    "Official_signaler": "others",
}


# input and output pkl files for training set
input_pkl = 'xxx/data/argo2/processed/orignal_pkls/argo2_infos_train.pkl'
output_pkl = 'xxx/data/argo2/processed/argo2_infos_train.pkl'

with open(input_pkl, 'rb') as f:
    data_list = pickle.load(f)

print(f"Loaded {len(data_list)} entries.")

for i, item in enumerate(data_list):
    assert 'name' in item['annos'], 'no name on %dth sample' % i
    name = item['annos']['name']  

    new_name = np.array([mapping.get(name, name) for name in name])

    item['annos']['name'] = new_name

with open(output_pkl, 'wb') as f:
    pickle.dump(data_list, f)

print(f"Saved updated list to {output_pkl}.")



# input and output pkl files for validation set
input_pkl = 'xxx/data/argo2/processed/orignal_pkls/argo2_infos_val.pkl'
output_pkl = 'xxx/data/argo2/processed/argo2_infos_val.pkl'

with open(input_pkl, 'rb') as f:
    data_list = pickle.load(f)

print(f"Loaded {len(data_list)} entries.")

for i, item in enumerate(data_list):
    assert 'name' in item['annos'], 'no name on %dth sample' % i
    name = item['annos']['name']  

    new_name = np.array([mapping.get(name, name) for name in name])

    item['annos']['name'] = new_name

with open(output_pkl, 'wb') as f:
    pickle.dump(data_list, f)

print(f"Saved updated list to {output_pkl}.")
