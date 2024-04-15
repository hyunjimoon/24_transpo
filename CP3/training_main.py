from __future__ import absolute_import
from __future__ import print_function

import datetime
import argparse
from training_simulation import Simulation
from generator import TrafficGenerator
from replay_buffer import ReplayBuffer
from model import DQN
from utils import import_train_configuration, set_sumo, set_train_path
import torch
import pandas as pd
import wandb

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
print('Device', DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--flow', type=int, default=1000, help='Flow of cars')
    parser.add_argument('--lane', type=float, default=4.0, help='Number of lanes')
    parser.add_argument('--length', type=float, default=750, help='Length of lanes')
    parser.add_argument('--speed', type=float, default=13.89, help='Speed limit')
    parser.add_argument('--left', type=float, default=0.25, help='Left turn ratio')
    parser.add_argument('--reward', type=str, default='base', help='Choose among different reward types: base, waittime, speed, and custom')
    args = parser.parse_args()
    
    # import default config and init config
    config = import_train_configuration(config_file='settings/training_settings.ini')
    
    # update config with the arguments
    config['n_cars_generated'] = args.flow
    config['num_lanes'] = args.lane
    config['lane_length'] = args.length
    config['speed_limit'] = args.speed
    config['left_turn'] = args.left
    config['reward_type'] = args.reward

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['num_lanes'],
        config['lane_length'],
        config['speed_limit'],
        config['left_turn']
    )
    
    dir_name = f"intersection_reward-{config['reward_type']}_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}"
    sumo_network_dir_name = f"intersection_flow{config['n_cars_generated']}_lane{config['num_lanes']}_length{config['lane_length']}_speed{config['speed_limit']}_left{config['left_turn']}"
    sumo_cmd = set_sumo(config['gui'], sumo_network_dir_name, config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['path_name']+"/"+dir_name)

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )
    
    DQN.to(DEVICE)
    
    ReplyBuffer = ReplayBuffer(
        config['memory_size_max'], 
        config['memory_size_min']
    )
        
    Simulation = Simulation(
        DQN,
        ReplyBuffer,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['batch_size'],
        config['learning_rate'],
        config['num_lanes'],
        config['lane_length'],
        config['reward_type']
        
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()

    project = "DQN ATL"
    if config['wandb'] == 'True':
        wandb.init(project=project, name=dir_name, config=config)
        
    result_save = pd.DataFrame(columns=['episode', 'avg_reward', 'avg_waiting', 'avg_average_speed', 'avg_current_veh', 'avg_passed_veh', 'training_time', 'simulation_time'])

    while episode < config['total_episodes']:
        print('\n [INFO]----- Episode', str(episode + 1), '/', str(config['total_episodes']), '-----')
        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / config['total_episodes'])
        # run the simulation
        simulation_time, training_time, avg_reward, avg_waiting, training_loss, avg_average_speed, avg_current_veh, avg_passed_veh = Simulation.run(episode, epsilon)
            
        print('\t [STAT] Simulation time:', simulation_time, 's - Training time:',
              training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
        if config['wandb'] == 'True':
            # log the training progress in wandb
            wandb.log({
                "all/training_loss": training_loss,
                "all/avg_reward": avg_reward,
                "all/avg_waiting_time": avg_waiting,
                "all/avg_average_speed": avg_average_speed,
                "all/avg_current_vehicles": avg_current_veh,
                "all/avg_passed_vehicles": avg_passed_veh,
                "all/simulation_time": simulation_time,
                "all/training_time": training_time,
                "all/entropy": epsilon}, step=episode)
        # save the results
        new_row = pd.DataFrame({'episode': [episode], 
                                'avg_reward': [avg_reward], 
                                'avg_waiting': [avg_waiting], 
                                'avg_average_speed': [avg_average_speed], 
                                'avg_current_veh': [avg_current_veh], 
                                'avg_passed_veh': [avg_passed_veh], 
                                'training_time': [training_time], 
                                'simulation_time': [simulation_time]})

        result_save = pd.concat([result_save, new_row], ignore_index=True)

        episode += 1
        print('\t [INFO] Saving the model')
        Simulation.save_model(path, episode)
    
    result_save.to_csv(path + '/training_results.csv', index=False)

    print("\n [INFO] End of Training")
    print("\t [STAT] Start time:", timestamp_start)
    print("\t [STAT] End time:", datetime.datetime.now())
    print("\t [STAT] Session info saved at:", path)
