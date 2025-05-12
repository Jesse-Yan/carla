import carla
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

try:
    client = carla.Client('localhost', 2000) # 如果您的服务器不在本地或端口不同，请修改
    client.set_timeout(10.0)
    world = client.get_world()

    logging.info("Querying available vehicle blueprints...")
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')

    if not vehicle_blueprints:
        logging.warning("No vehicle blueprints found on the server!")
    else:
        logging.info("Available vehicle blueprints:")
        for bp in sorted(vehicle_blueprints, key=lambda x: x.id): # 按ID排序方便查看
            logging.info(f"  - {bp.id}")

    # (可选) 只列出明确标记为 'car' 类型的车辆
    # logging.info("\nAvailable 'car' type blueprints (base_type='car'):")
    # car_blueprints = [bp for bp in vehicle_blueprints if bp.has_attribute('base_type') and bp.get_attribute('base_type') == 'car']
    # if car_blueprints:
    #     for bp in sorted(car_blueprints, key=lambda x: x.id):
    #         logging.info(f"  - {bp.id} (Generations: {bp.get_attribute('generation').as_str() if bp.has_attribute('generation') else 'N/A'})")
    # else:
    #     logging.warning("No blueprints explicitly identified as 'base_type' = 'car'.")


except Exception as e:
    logging.critical(f"An error occurred: {e}")

finally:
    logging.info("Script finished.")