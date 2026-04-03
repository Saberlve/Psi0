import argparse

from manager import TeleopManager

# create a tv.step() thread and request image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument(
        "--task_name", type=str, default="default_task", help="Name of the task"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--robot", default="g1", help="Use g1 controllers")
    parser.add_argument("--pico_streamer", action="store_true", help="Enable Pico IR Streamer")
    parser.add_argument("--pico_ip", type=str, default="192.168.0.128", help="Pico IP address")
    args = parser.parse_args()

    manager = TeleopManager(
        task_name=args.task_name, robot=args.robot, debug=args.debug, pico_streamer=args.pico_streamer, pico_ip=args.pico_ip
    )
    manager.start_processes()
    # TODO: run in two separate terminals for debuggnig
    manager.run_command_loop()
