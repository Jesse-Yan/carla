
"""
Generates a dataset from CARLA. Spawns a total number of vehicles,
designates a subset as 'ego' vehicles which are instrumented with multiple cameras
for RGB image and YAML metadata generation. Other vehicles act as background traffic.
Also spawns pedestrians.
Output format: <base_folder>/<scenario_name>/<ego_vehicle_id>/<timestamp>.yaml and images 
(RGB: <timestamp>_<camera_name>.png; BEV: <timestamp>_bev.png).
6D Pose order: [x, y, z, roll, pitch, yaw].
"""

import time
import carla
import argparse
import logging
import os
import numpy as np
import math
import yaml # PyYAML: pip install PyYAML
from numpy import random as np_random
from carla import ColorConverter # For BEV coloring (optional)

# --- 默认配置 ---
DEFAULT_IMAGES_PER_CAMERA = 10 # Applies to RGB cameras
DEFAULT_BASE_DATASET_FOLDER = "carla_dataset"
DEFAULT_SCENARIO_NAME_PREFIX = "scenario"
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720
DEFAULT_CAMERA_FOV = 110.0 
DEFAULT_DELTA_SECONDS = 0.1

DEFAULT_BEV_IMAGE_WIDTH = 400 
DEFAULT_BEV_IMAGE_HEIGHT = 400
DEFAULT_BEV_FOV = 90.0
DEFAULT_BEV_Z_OFFSET = 15.0 

# 全局字典和列表
capture_counts = {}
sensor_actors_list = [] 
vehicle_camera_configs_map = {} 
all_actor_ids_for_cleanup = []
walkers_list_managed = []
pedestrian_ai_controllers_list = []

# --- YAML Dumper ---
class PrettySafeDumper(yaml.SafeDumper):
    def represent_float(self, data):
        if data != data or data == float('inf') or data == float('-inf'):
            return self.represent_scalar('tag:yaml.org,2002:float', str(data))
        return super().represent_float(data)
    def represent_list(self, data):
        is_matrix_like = False
        if isinstance(data, list) and len(data) > 0:
            if all(isinstance(row, list) and len(row) > 0 and all(isinstance(el, (float, int)) for el in row) for row in data):
                is_matrix_like = True
            elif all(isinstance(el, (float, int)) for el in data) and len(data) > 3:
                is_matrix_like = True
        if is_matrix_like:
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        return super().represent_list(data)

yaml.SafeDumper.add_representer(float, PrettySafeDumper.represent_float)
yaml.SafeDumper.add_representer(list, PrettySafeDumper.represent_list)

# --- 辅助函数 ---

def carla_transform_to_numpy_matrix(transform: carla.Transform) -> np.ndarray:
    """Converts a carla.Transform to a 4x4 numpy matrix."""
    matrix_list_of_lists = transform.get_matrix() # carla.Transform.get_matrix() 返回 4x4 列表的列表
    return np.array(matrix_list_of_lists)

def carla_transform_to_6d_pose_list(transform: carla.Transform) -> list:
    loc = transform.location
    rot = transform.rotation
    return [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw] # User confirmed final order

def calculate_intrinsic_matrix(image_w: int, image_h: int, fov_degrees: float) -> np.ndarray:
    fov_rad = fov_degrees * (math.pi / 180.0)
    fx = image_w / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx 
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return K

def get_actor_blueprints(world: carla.World, filter_pattern: str, generation_str: str) -> list:
    blueprints = world.get_blueprint_library().filter(filter_pattern)
    if not blueprints: logging.warning(f"No BPs for filter '{filter_pattern}'."); return []
    if generation_str.lower() == "all" or len(blueprints) == 1: return list(blueprints)
    try:
        is_vehicle_filter = "vehicle" in filter_pattern.lower()
        valid_generations = [1, 2] if is_vehicle_filter else []
        if not valid_generations and is_vehicle_filter: return list(blueprints)
        int_generation = int(generation_str)
        if int_generation in valid_generations:
            selected_bps = [bp for bp in blueprints if bp.has_attribute('generation') and bp.get_attribute('generation').as_int() == int_generation]
            if not selected_bps and is_vehicle_filter: return list(blueprints)
            return selected_bps if selected_bps else list(blueprints)
        else: return list(blueprints)
    except: return list(blueprints)


def process_rgb_image_for_dataset(image_data: carla.Image, base_scenario_output_path: str, 
                                 vehicle_id_str: str, camera_actor_id_str: str, 
                                 camera_name_for_file: str, images_to_capture_per_cam: int):
    unique_camera_key_for_counting = f"v{vehicle_id_str}_c{camera_actor_id_str}"
    if unique_camera_key_for_counting not in capture_counts:
        capture_counts[unique_camera_key_for_counting] = 0
    if capture_counts[unique_camera_key_for_counting] < images_to_capture_per_cam:
        current_frame_str = f"{image_data.frame:06d}"
        vehicle_specific_output_path = os.path.join(base_scenario_output_path, vehicle_id_str)
        image_filename = f"{current_frame_str}_{camera_name_for_file}.png"
        full_image_file_path = os.path.join(vehicle_specific_output_path, image_filename)
        try:
            image_data.save_to_disk(full_image_file_path)
            capture_counts[unique_camera_key_for_counting] += 1
        except Exception as e: logging.error(f"Err saving RGB img {full_image_file_path}: {e}")


def process_bev_image_for_dataset(semantic_image_data: carla.Image, 
                                  base_scenario_output_path: str, 
                                  vehicle_id_str: str):
    current_frame_str = f"{semantic_image_data.frame:06d}"
    vehicle_specific_output_path = os.path.join(base_scenario_output_path, vehicle_id_str)
    os.makedirs(vehicle_specific_output_path, exist_ok=True)
    bev_image_filename = f"{current_frame_str}_bev.png" 
    full_bev_image_path = os.path.join(vehicle_specific_output_path, bev_image_filename)
    try:
        # semantic_image_data.convert(ColorConverter.CityScapesPalette) # Optional: colorize
        semantic_image_data.save_to_disk(full_bev_image_path)
    except Exception as e: logging.error(f"Error saving BEV image {full_bev_image_path}: {e}")


def write_vehicle_yaml_for_frame(base_scenario_output_path: str, vehicle_actor: carla.Actor, 
                                 frame_id: int, camera_configs_for_this_vehicle: list,
                                 world_ref: carla.World):
    vehicle_id_str = str(vehicle_actor.id)
    current_timestamp_str = f"{frame_id:06d}"
    vehicle_specific_output_path = os.path.join(base_scenario_output_path, vehicle_id_str)
    os.makedirs(vehicle_specific_output_path, exist_ok=True)
    yaml_file_path = os.path.join(vehicle_specific_output_path, f"{current_timestamp_str}.yaml")

    vehicle_world_transform_carla = vehicle_actor.get_transform()
    vehicle_6d_pose_world = carla_transform_to_6d_pose_list(vehicle_world_transform_carla)
    velocity_vec = vehicle_actor.get_velocity()
    ego_speed_kmh = 3.6 * math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)

    yaml_data = {
        "lidar_pose": vehicle_6d_pose_world, "true_ego_pos": vehicle_6d_pose_world,
        "ego_speed": round(ego_speed_kmh, 2)
    }
    for cam_conf_entry in camera_configs_for_this_vehicle:
        cam_yaml_id = cam_conf_entry["id_in_yaml"]
        camera_actor = world_ref.get_actor(cam_conf_entry["actor_id"])
        if not camera_actor or not camera_actor.is_alive: continue
        cam_world_transform_carla = camera_actor.get_transform()
        cam_6d_world_cords = carla_transform_to_6d_pose_list(cam_world_transform_carla)
        yaml_data[cam_yaml_id] = {
            "cords": cam_6d_world_cords,
            "extrinsic": cam_conf_entry["relative_transform_matrix"].tolist(),
            "intrinsic": cam_conf_entry["K_matrix"].tolist(),
            "image_size": cam_conf_entry["image_size_xy"],
            "distortion_coefficients": cam_conf_entry["distortion_coefficients"]
        }
    try:
        with open(yaml_file_path, 'w') as f:
            yaml.dump(yaml_data, f, indent=2, sort_keys=False, Dumper=PrettySafeDumper, default_flow_style=None)
    except Exception as e: logging.error(f"Err writing YAML {yaml_file_path}: {e}")

# --- 主函数 ---
def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    # CARLA Connection & Simulation
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA Simulator host IP')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA Simulator TCP port')
    argparser.add_argument('--tm-port', default=8000, type=int, help='Traffic Manager port')
    argparser.add_argument('--asynch', action='store_true', help='Run in asynchronous mode (default: synchronous)')
    argparser.add_argument('--delta-seconds', type=float, default=DEFAULT_DELTA_SECONDS, help='Fixed delta seconds for synchronous mode')
    argparser.add_argument('-s', '--seed', type=int, default=None, help='Global random seed for CARLA, TM, numpy.random')
    
    # Dataset Output
    argparser.add_argument('--base-dataset-folder', type=str, default=DEFAULT_BASE_DATASET_FOLDER, help='Base directory for the generated dataset')
    argparser.add_argument('--scenario-name', type=str, default=None, help='Name for the scenario subfolder (default: "scenario_YYYYMMDD_HHMMSS")')

    # Vehicle Configuration
    argparser.add_argument('--total-vehicles', metavar='N_TOTAL', default=10, type=int, help='Total number of vehicles to spawn (background + ego)')
    argparser.add_argument('--num-ego-vehicles', metavar='N_EGO', default=1, type=int, help='Number of ego vehicles to instrument with cameras and save data for')
    argparser.add_argument('--filterv', default='vehicle.tesla.model3,vehicle.audi.etron,vehicle.bmw.grandtourer', help='Comma-separated vehicle BPs or filter pattern')
    argparser.add_argument('--generationv', default='All', help='Vehicle generation filter ("1", "2", "All")')
    argparser.add_argument('--safe', action='store_true', help='Spawn only "car" type vehicles if filterv is generic')
    argparser.add_argument('--car-lights-on', action='store_true', help='Enable automatic car light management for all vehicles')

    # Pedestrian Configuration
    argparser.add_argument('-w', '--number-of-walkers', default=20, type=int, help='Number of walkers')
    argparser.add_argument('--filterw', default='walker.pedestrian.*', help='Filter for walker blueprints')
    argparser.add_argument('--generationw', default='All', help='Walker generation filter (typically "All")')
    argparser.add_argument('--seedw', type=int, default=None, help='Specific seed for pedestrian spawning (defaults to main seed if not set)')
    argparser.add_argument('--pedestrian-crossing-percentage', type=float, default=0.1, help='Percentage of peds that cross roads')
    argparser.add_argument('--pedestrian-running-percentage', type=float, default=0.1, help='Percentage of peds that run')

    # Camera & Image Configuration (applies to ego vehicles)
    argparser.add_argument('--images-per-rgb-camera', type=int, default=DEFAULT_IMAGES_PER_CAMERA, help='Images to save per RGB camera, per EGO vehicle')
    argparser.add_argument('--image-width', type=int, default=DEFAULT_IMAGE_WIDTH, help='Width of RGB camera images')
    argparser.add_argument('--image-height', type=int, default=DEFAULT_IMAGE_HEIGHT, help='Height of RGB camera images')
    argparser.add_argument('--camera-fov', type=float, default=DEFAULT_CAMERA_FOV, help='RGB Camera Horizontal Field of View in degrees')
    
    # BEV Camera Configuration (applies to ego vehicles)
    argparser.add_argument('--bev-width', type=int, default=DEFAULT_BEV_IMAGE_WIDTH, help='Width of BEV images')
    argparser.add_argument('--bev-height', type=int, default=DEFAULT_BEV_IMAGE_HEIGHT, help='Height of BEV images')
    argparser.add_argument('--bev-fov', type=float, default=DEFAULT_BEV_FOV, help='BEV Camera Field of View')
    argparser.add_argument('--bev-z-offset', type=float, default=DEFAULT_BEV_Z_OFFSET, help='BEV Camera height above vehicle')
    
    argparser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')

    args = argparser.parse_args()

    if args.num_ego_vehicles > args.total_vehicles:
        logging.warning("--num-ego-vehicles cannot exceed --total-vehicles. Setting num_ego_vehicles = total_vehicles.")
        args.num_ego_vehicles = args.total_vehicles
    if args.num_ego_vehicles < 0: args.num_ego_vehicles = 0
    if args.total_vehicles < 0: args.total_vehicles = 0

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    if args.seed is not None: np_random.seed(args.seed); random.seed(args.seed) 
    
    scenario_name_to_use = args.scenario_name if args.scenario_name else f"{DEFAULT_SCENARIO_NAME_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    current_scenario_output_path = os.path.join(args.base_dataset_folder, scenario_name_to_use)
    try:
        os.makedirs(current_scenario_output_path, exist_ok=True)
        logging.info(f"Dataset will be saved in: {current_scenario_output_path}")
    except OSError as e: logging.critical(f"Could not create base dir: {e}"); return
    
    # These are now local to main, will be populated.
    # Initializing them as empty lists here.
    python_list_all_spawned_vehicles = [] 
    ego_vehicle_actor_objects = []  
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = None
    original_settings = None
    synchronous_master = False

    try:
        world = client.get_world()
        original_settings = world.get_settings()
        current_map = world.get_map()
        logging.info(f"Connected. Map: {current_map.name}")

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.set_hybrid_physics_mode(args.hybrid)
        if args.hybrid: traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None: traffic_manager.set_random_device_seed(args.seed)
        
        current_sim_settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not current_sim_settings.synchronous_mode:
                synchronous_master = True
                current_sim_settings.synchronous_mode = True
                current_sim_settings.fixed_delta_seconds = args.delta_seconds
            else: 
                synchronous_master = False
                if current_sim_settings.fixed_delta_seconds is None: current_sim_settings.fixed_delta_seconds = args.delta_seconds
                if abs(current_sim_settings.fixed_delta_seconds - args.delta_seconds) > 0.001 and not synchronous_master : # Corrected logic for warning
                     logging.warning(f"World already sync with delta_s={current_sim_settings.fixed_delta_seconds}. Script's delta_s={args.delta_seconds} will be used by TM if this script is not the tick master.")
        else:
            current_sim_settings.synchronous_mode = False; current_sim_settings.fixed_delta_seconds = None
            traffic_manager.set_synchronous_mode(False)
        world.apply_settings(current_sim_settings)
        actual_delta_seconds = current_sim_settings.fixed_delta_seconds if current_sim_settings.synchronous_mode and current_sim_settings.fixed_delta_seconds is not None else (1.0/20.0)
        logging.info(f"World settings: Sync={current_sim_settings.synchronous_mode}, DeltaSec={actual_delta_seconds}")

        # --- Vehicle Spawning (Total Vehicles) ---
        vehicle_bps_filter_str = args.filterv
        if ',' in vehicle_bps_filter_str:
            vehicle_names_list = [name.strip() for name in vehicle_bps_filter_str.split(',') if name.strip()]
            vehicle_blueprint_list = []
            for name in vehicle_names_list:
                try: vehicle_blueprint_list.append(world.get_blueprint_library().find(name))
                except IndexError: logging.warning(f"Vehicle BP '{name}' not found. Skipping.")
        else:
            vehicle_blueprint_list = get_actor_blueprints(world, vehicle_bps_filter_str, args.generationv)

        if not vehicle_blueprint_list: raise ValueError("No vehicle blueprints found after filtering.")
        if args.safe:
            vehicle_blueprint_list = [bp for bp in vehicle_blueprint_list if bp.get_attribute('base_type') == 'car']
            if not vehicle_blueprint_list: raise ValueError("Safe mode: No 'car' BPs found.")
        
        vehicle_blueprint_list = sorted(vehicle_blueprint_list, key=lambda bp: bp.id)
        if not vehicle_blueprint_list: raise ValueError("No vehicle blueprints available after sorting/filtering.")

        available_spawn_points = current_map.get_spawn_points()
        num_available_spawn_points = len(available_spawn_points)
        if args.total_vehicles > num_available_spawn_points:
            logging.warning(f"Requested {args.total_vehicles} total vehicles, but map only has {num_available_spawn_points} spawn points. Reducing to {num_available_spawn_points}.")
            args.total_vehicles = num_available_spawn_points
        
        temp_spawned_vehicle_actor_ids = []
        if args.total_vehicles > 0:
            np_random.shuffle(available_spawn_points)
            batch_cmds_spawn_all_vehicles = []
            for i in range(args.total_vehicles):
                blueprint = np_random.choice(vehicle_blueprint_list)
                if blueprint.has_attribute('color'): blueprint.set_attribute('color', np_random.choice(blueprint.get_attribute('color').recommended_values))
                if blueprint.has_attribute('driver_id'): blueprint.set_attribute('driver_id', np_random.choice(blueprint.get_attribute('driver_id').recommended_values))
                blueprint.set_attribute('role_name', 'autopilot')
                
                transform = available_spawn_points[i]
                batch_cmds_spawn_all_vehicles.append(carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))
            
            responses = client.apply_batch_sync(batch_cmds_spawn_all_vehicles, synchronous_master and current_sim_settings.synchronous_mode)
            for response in responses:
                if response.error: logging.error(f"  Error spawning a vehicle: {response.error}")
                else:
                    temp_spawned_vehicle_actor_ids.append(response.actor_id)
                    all_actor_ids_for_cleanup.append(response.actor_id)
            logging.info(f"Spawned {len(temp_spawned_vehicle_actor_ids)} total vehicles.")
        
        # ** FIX for ActorList slicing and clear **
        carla_list_all_vehicles_temp = world.get_actors(temp_spawned_vehicle_actor_ids)
        python_list_all_spawned_vehicles = list(carla_list_all_vehicles_temp) # Convert to Python list
        
        # --- Select Ego Vehicles and Instrument Them ---
        num_ego_to_instrument = min(args.num_ego_vehicles, len(python_list_all_spawned_vehicles))
        ego_vehicle_actor_objects = python_list_all_spawned_vehicles[:num_ego_to_instrument] # Slice Python list
        
        logging.info(f"Instrumenting {len(ego_vehicle_actor_objects)} EGO vehicles with sensors.")

        if ego_vehicle_actor_objects: # Check if there are any ego vehicles to instrument
            rgb_camera_setups_definition = [
                {"id_in_yaml": "camera0", "transform": carla.Transform(carla.Location(x=2.0, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=0, roll=0))},
                {"id_in_yaml": "camera1", "transform": carla.Transform(carla.Location(x=-2.5, y=0.0, z=1.4), carla.Rotation(pitch=0, yaw=180, roll=0))},
                {"id_in_yaml": "camera2", "transform": carla.Transform(carla.Location(x=0.5, y=-1.3, z=1.3), carla.Rotation(pitch=0, yaw=-75, roll=0))},
                {"id_in_yaml": "camera3", "transform": carla.Transform(carla.Location(x=0.5, y=1.3, z=1.3), carla.Rotation(pitch=0, yaw=75, roll=0))}
            ]
            base_rgb_cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            base_rgb_cam_bp.set_attribute('image_size_x', str(args.image_width))
            base_rgb_cam_bp.set_attribute('image_size_y', str(args.image_height))
            base_rgb_cam_bp.set_attribute('fov', str(args.camera_fov))
            if current_sim_settings.synchronous_mode and actual_delta_seconds > 0:
                base_rgb_cam_bp.set_attribute('sensor_tick', str(actual_delta_seconds))
            
            calculated_rgb_k_matrix = calculate_intrinsic_matrix(args.image_width, args.image_height, args.camera_fov)
            default_distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

            bev_cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            bev_cam_bp.set_attribute('image_size_x', str(args.bev_width))
            bev_cam_bp.set_attribute('image_size_y', str(args.bev_height))
            bev_cam_bp.set_attribute('fov', str(args.bev_fov))
            if current_sim_settings.synchronous_mode and actual_delta_seconds > 0:
                bev_cam_bp.set_attribute('sensor_tick', str(actual_delta_seconds))
            bev_cam_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=args.bev_z_offset), carla.Rotation(pitch=-90, yaw=0, roll=0))

            for veh_actor in ego_vehicle_actor_objects:
                if not veh_actor.is_alive: continue
                vehicle_id_str_for_path = str(veh_actor.id)
                vehicle_data_path_for_ego = os.path.join(current_scenario_output_path, vehicle_id_str_for_path)
                os.makedirs(vehicle_data_path_for_ego, exist_ok=True)
                
                current_vehicle_cam_cfgs_for_yaml = []
                logging.info(f"  Attaching RGB cameras to EGO vehicle {veh_actor.id}:")
                for cam_setup_def in rgb_camera_setups_definition:
                    cam_actor = world.try_spawn_actor(base_rgb_cam_bp, cam_setup_def["transform"], attach_to=veh_actor)
                    if cam_actor:
                        all_actor_ids_for_cleanup.append(cam_actor.id)
                        sensor_actors_list.append(cam_actor)
                        current_vehicle_cam_cfgs_for_yaml.append({
                            "id_in_yaml": cam_setup_def["id_in_yaml"], "actor_id": cam_actor.id,
                            "relative_transform_matrix": carla_transform_to_numpy_matrix(cam_setup_def["transform"]),
                            "K_matrix": calculated_rgb_k_matrix,
                            "distortion_coefficients": default_distortion_coeffs,
                            "image_size_xy": [args.image_width, args.image_height]
                        })
                        cam_actor.listen(lambda image_data, v_id_str=str(veh_actor.id), c_id_str=str(cam_actor.id), 
                                                        c_name_file=cam_setup_def["id_in_yaml"]:
                                         process_rgb_image_for_dataset(image_data, current_scenario_output_path, 
                                                                   v_id_str, c_id_str, c_name_file, args.images_per_rgb_camera))
                        logging.info(f"    Attached RGB camera '{cam_setup_def['id_in_yaml']}' (ID: {cam_actor.id})")
                    else: logging.warning(f"    Failed to attach RGB camera '{cam_setup_def['id_in_yaml']}' to EGO vehicle {veh_actor.id}")
                
                logging.info(f"  Attaching BEV camera to EGO vehicle {veh_actor.id}:")
                bev_cam_actor = world.try_spawn_actor(bev_cam_bp, bev_cam_transform, attach_to=veh_actor)
                if bev_cam_actor:
                    all_actor_ids_for_cleanup.append(bev_cam_actor.id)
                    # BEV sensor is not added to sensor_actors_list for separate capture count target
                    bev_cam_actor.listen(lambda sem_seg_data, v_id_str=str(veh_actor.id):
                                         process_bev_image_for_dataset(sem_seg_data, current_scenario_output_path, v_id_str))
                    logging.info(f"    Attached BEV camera (ID: {bev_cam_actor.id})")
                else: logging.warning(f"    Failed to attach BEV camera to EGO vehicle {veh_actor.id}")
                vehicle_camera_configs_map[veh_actor.id] = current_vehicle_cam_cfgs_for_yaml
        
        if not sensor_actors_list and args.num_ego_vehicles > 0 :
             logging.warning("No RGB cameras were successfully attached to any EGO vehicle.")
        
        if args.car_lights_on:
            for v_actor in python_list_all_spawned_vehicles: # Use the Python list for iteration
                if v_actor.is_alive: traffic_manager.update_vehicle_lights(v_actor, True)

        # --- Pedestrian Spawning ---
        if args.number_of_walkers > 0:
            # ... (Pedestrian spawning logic remains the same as your last full script) ...
            logging.info(f"Spawning {args.number_of_walkers} walkers...")
            walker_seed_to_use = args.seedw if args.seedw is not None else args.seed
            if walker_seed_to_use is not None: world.set_pedestrians_seed(walker_seed_to_use); np_random.seed(walker_seed_to_use)
            walker_bps_list = get_actor_blueprints(world, args.filterw, args.generationw)
            if not walker_bps_list: logging.warning("No walker BPs. No walkers spawned.")
            else:
                spawn_points_walkers = []; num_retries = args.number_of_walkers * 3
                for _ in range(num_retries): 
                    if len(spawn_points_walkers) >= args.number_of_walkers: break
                    wp_transform = carla.Transform(); wp_loc = world.get_random_location_from_navigation()
                    if wp_loc: wp_transform.location = wp_loc; wp_transform.location.z += 1.0; spawn_points_walkers.append(wp_transform)
                
                num_walkers_to_spawn = min(args.number_of_walkers, len(spawn_points_walkers))
                if num_walkers_to_spawn < args.number_of_walkers: logging.warning(f"Walkers: {num_walkers_to_spawn} < Req.")

                batch_cmds_walkers, speeds = [], []
                for i in range(num_walkers_to_spawn):
                    bp_w = np_random.choice(walker_bps_list)
                    if bp_w.has_attribute('is_invincible'): bp_w.set_attribute('is_invincible', 'false')
                    s_attr = bp_w.get_attribute('speed')
                    s = (s_attr.recommended_values[1] if np_random.random() > args.pedestrian_running_percentage else s_attr.recommended_values[2]) if s_attr and len(s_attr.recommended_values) > 2 else 1.4
                    speeds.append(s); batch_cmds_walkers.append(carla.command.SpawnActor(bp_w, spawn_points_walkers[i]))
                
                res_w = client.apply_batch_sync(batch_cmds_walkers, True)
                tmp_w_data = []
                for i, r in enumerate(res_w):
                    if r.error: logging.error(f" Spawn walker err: {r.error}")
                    else: tmp_w_data.append({"id":r.actor_id, "speed":speeds[i]}); all_actor_ids_for_cleanup.append(r.actor_id)
                
                batch_ctrl = []; bp_ctrl = world.get_blueprint_library().find('controller.ai.walker')
                for d_w in tmp_w_data: batch_ctrl.append(carla.command.SpawnActor(bp_ctrl, carla.Transform(), d_w["id"]))
                res_ctrl = client.apply_batch_sync(batch_ctrl, True)
                for i, r in enumerate(res_ctrl):
                    if r.error: logging.error(f" Spawn AI ctrl err: {r.error}")
                    else:
                        walkers_list_managed.append({"id":tmp_w_data[i]["id"], "con":r.actor_id, "speed":tmp_w_data[i]["speed"]})
                        all_actor_ids_for_cleanup.append(r.actor_id)
                
                if current_sim_settings.synchronous_mode and synchronous_master: world.tick()
                else: world.wait_for_tick()
                world.set_pedestrians_cross_factor(args.pedestrian_crossing_percentage)
                for mw_info in walkers_list_managed:
                    ctrl_actor = world.get_actor(mw_info["con"])
                    if ctrl_actor and ctrl_actor.is_alive:
                        pedestrian_ai_controllers_list.append(ctrl_actor)
                        ctrl_actor.start(); ctrl_actor.go_to_location(world.get_random_location_from_navigation()); ctrl_actor.set_max_speed(float(mw_info["speed"]))
                logging.info(f"Spawned and init {len(walkers_list_managed)} walkers.")
            if args.seed is not None and (args.seedw is not None and walker_seed_to_use != args.seed) :
                np_random.seed(args.seed); random.seed(args.seed)


        # --- Main Simulation Loop ---
        logging.info(f"Sim start. Target: {args.images_per_rgb_camera} RGB images for each of {len(sensor_actors_list)} EGO cameras. BEV images saved per frame.")
        sim_frames_elapsed = 0
        sim_fps_effective = 1.0 / actual_delta_seconds if actual_delta_seconds > 0 else 10.0
        
        if not ego_vehicle_actor_objects: 
             max_simulation_frames = int(sim_fps_effective * 30)
             logging.info(f"No EGO vehicles. Sim will run for approx {max_simulation_frames} frames (30s).")
        elif not sensor_actors_list: 
             max_simulation_frames = int(sim_fps_effective * 30) 
             logging.info(f"EGO vehicles present, but no RGB cameras. Sim will run for approx {max_simulation_frames} frames (30s).")
        else: 
             max_simulation_frames = int(args.images_per_rgb_camera * 1.5) + int(sim_fps_effective * 30)
             logging.info(f"Max simulation frames for data collection: {max_simulation_frames}")

        simulation_start_wall_time = time.time()
        last_yaml_written_server_frame = -1 

        while True:
            if current_sim_settings.synchronous_mode and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
            
            current_server_frame = world.get_snapshot().frame
            sim_frames_elapsed += 1

            if ego_vehicle_actor_objects and current_server_frame > last_yaml_written_server_frame:
                for ego_veh_actor in ego_vehicle_actor_objects:
                    if ego_veh_actor.is_alive:
                        cam_cfgs = vehicle_camera_configs_map.get(ego_veh_actor.id, [])
                        if cam_cfgs: 
                             write_vehicle_yaml_for_frame(current_scenario_output_path, ego_veh_actor, 
                                                         current_server_frame, cam_cfgs, world)
                last_yaml_written_server_frame = current_server_frame
            
            if not sensor_actors_list: 
                if sim_frames_elapsed >= max_simulation_frames: break
                if sim_frames_elapsed % int(sim_fps_effective * 10) == 0: logging.info(f"  Simulated {sim_frames_elapsed} frames (no RGB cameras on EGO)...")
                continue

            num_rgb_cameras_finished_capture = 0
            for sensor in sensor_actors_list: 
                unique_cam_key = f"v{sensor.parent.id}_c{sensor.id}" if sensor.parent else None
                if not sensor.is_alive or (sensor.parent and not sensor.parent.is_alive):
                    num_rgb_cameras_finished_capture += 1; continue
                if unique_cam_key and capture_counts.get(unique_cam_key, 0) >= args.images_per_rgb_camera:
                    num_rgb_cameras_finished_capture += 1
            
            if num_rgb_cameras_finished_capture >= len(sensor_actors_list):
                logging.info("All EGO RGB cameras have captured target images or are no longer active.")
                break

            if sim_frames_elapsed % int(sim_fps_effective * 10) == 0:
                logging.info(f"  Simulated {sim_frames_elapsed} frames ({current_server_frame} server frame) ({time.time() - simulation_start_wall_time:.1f}s wall time). "
                             f"EGO RGB Cams done: {num_rgb_cameras_finished_capture}/{len(sensor_actors_list)}. Counts: {capture_counts}")
            
            if sim_frames_elapsed > max_simulation_frames:
                logging.warning(f"Max sim frames ({max_simulation_frames}) reached. Stopping. Counts: {capture_counts}")
                break
        
        logging.info(f"Data collection loop finished. Wall time: {time.time() - simulation_start_wall_time:.2f}s for {sim_frames_elapsed} script ticks.")

    except ValueError as ve: logging.critical(f"ValueError: {ve}")
    except RuntimeError as re: logging.critical(f"CARLA RuntimeError: {re}")
    except Exception as e: logging.critical(f"Unexpected error: {e}"); import traceback; traceback.print_exc()
    finally:
        logging.info("\n--- Starting Cleanup ---")
        if world and original_settings:
            logging.info("  Restoring original world settings...")
            if synchronous_master and not args.asynch: 
                final_settings = world.get_settings()
                final_settings.synchronous_mode = False
                final_settings.fixed_delta_seconds = None
                world.apply_settings(final_settings)
        if 'traffic_manager' in locals() and traffic_manager: 
            traffic_manager.set_synchronous_mode(False)

        # Combine all sensor types that need stopping their listeners
        all_sensors_to_stop = sensor_actors_list[:] # RGB cameras
        # Find BEV cameras if they were spawned and need explicit stopping (they are in all_actor_ids_for_cleanup)
        # For simplicity, let's assume BEV sensors are also in a list if we need to stop them this way.
        # The current BEV setup doesn't add them to sensor_actors_list for counting.
        # A more robust way:
        if world: # Check if world object is available
            for actor_id_to_check in all_actor_ids_for_cleanup:
                actor_to_check = world.get_actor(actor_id_to_check)
                if actor_to_check and actor_to_check.is_alive and 'sensor.camera.semantic_segmentation' in actor_to_check.type_id:
                    if actor_to_check not in all_sensors_to_stop: # Avoid duplicates if it was somehow added
                        all_sensors_to_stop.append(actor_to_check)
        
        logging.info(f"  Stopping {len(all_sensors_to_stop)} sensors (RGB + BEV)...")
        for sensor in all_sensors_to_stop:
            if sensor.is_alive and sensor.is_listening: # Check is_alive too
                try: sensor.stop()
                except Exception as e: logging.warning(f"    Error stopping sensor {sensor.id}: {e}")
        
        logging.info(f"  Stopping {len(pedestrian_ai_controllers_list)} AI controllers...")
        for ctrl in pedestrian_ai_controllers_list:
            if ctrl.is_alive:
                try: ctrl.stop()
                except Exception as e: logging.warning(f"    Error stopping AI controller {ctrl.id}: {e}")

        logging.info(f"  Destroying {len(all_actor_ids_for_cleanup)} actor IDs...")
        if all_actor_ids_for_cleanup and client:
            destroy_cmds = [carla.command.DestroyActor(actor_id) for actor_id in all_actor_ids_for_cleanup]
            batch_destroy_size = 100 
            for i in range(0, len(destroy_cmds), batch_destroy_size):
                try: client.apply_batch_sync(destroy_cmds[i:i+batch_destroy_size], True)
                except Exception as e: logging.warning(f" Error during batch destroy: {e}")
            logging.info(f"    All destroy commands sent for {len(all_actor_ids_for_cleanup)} actor IDs.")
        else: logging.info("    No actors in cleanup list or client not available.")

        capture_counts.clear(); sensor_actors_list.clear(); vehicle_camera_configs_map.clear(); 
        walkers_list_managed.clear(); pedestrian_ai_controllers_list.clear(); all_actor_ids_for_cleanup.clear()
        # python_list_all_spawned_vehicles.clear() # These are local to main, no need to clear globally
        # ego_vehicle_actor_objects.clear()

        time.sleep(1.0)
        logging.info("--- Cleanup Finished ---")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt: logging.info('\nUser cancelled. Cleanup in main\'s finally.')
    except Exception as e: logging.critical(f"Critical error at __main__: {e}"); import traceback; traceback.print_exc()
    finally: logging.info('Script execution finalized.')
