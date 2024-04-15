from __future__ import absolute_import
from __future__ import print_function

import argparse
from testing_simulation import Simulation
from generator import TrafficGenerator
from model import DQN
import torch
from utils import import_test_configuration, set_sumo, set_test_path

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print('Device', DEVICE)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--flow', type=int, default=1000, help='Flow of cars')
    parser.add_argument('--lane', type=float, default=4.0, help='Number of lanes')
    parser.add_argument('--length', type=float, default=750, help='Length of lanes')
    parser.add_argument('--speed', type=float, default=13.89, help='Speed limit')
    parser.add_argument('--left', type=float, default=0.25, help='Left turn ratio')
    parser.add_argument('--reward', type=str, default='waittime', help='Choose among different reward types: waittime, speed, sparse, custom')
    args = parser.parse_args()
    
    config = import_test_configuration(config_file='settings/testing_settings.ini')
    
     # update config with the arguments
    config['n_cars_generated'] = args.flow
    config['num_lanes'] = args.lane
    config['lane_length'] = args.length
    config['speed_limit'] = args.speed
    config['left_turn'] = args.left
    config['reward_type'] = args.reward
    
    sumo_network_dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}"
    sumo_cmd = set_sumo(config['gui'], sumo_network_dir_name, config['sumocfg_file_name'], config['max_steps'])
    model_path = set_test_path(config['models_path_name'])

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'],
        output_dim=config['num_actions'],
        path=model_path,
        checkpoint=config['model_to_test']
    )
    DQN.to(DEVICE)

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['num_lanes'],
        config['lane_length'],
        config['speed_limit'],
        config['left_turn']
    )
        
    Simulation = Simulation(
        DQN,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions']
    )

    print('\n----- Test Episode')
    # run the simulation
    simulation_time, avg_reward, avg_waiting = Simulation.run(config['episode_seed'])
    print("\t [STAT] Average reward:", avg_reward,
          "Average waiting time:", avg_waiting,
          "Simulation time:", simulation_time, 's')
    print('\n----- End Test Episode')
