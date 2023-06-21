import argparse
import os
import random
import shutil
import string
import time

import numpy as np
import torch
from torchsummary import summary_string
from dataset import AudioFileDataset, LibriSpeechDataset
from model import SiameseSiren
from model_config import ModelConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


def find_value_for_string(string, search_string):
    return string[string.find(search_string)+len(search_string):string.find('\n', string.find(search_string))].replace(',', '')

def get_model_parameters(ts_result):
    total_params = find_value_for_string(ts_result, 'Total params: ')
    trainable_params = find_value_for_string(ts_result, 'Trainable params: ')
    non_trainable_params = find_value_for_string(ts_result, 'Non-trainable params: ')
    params_size = find_value_for_string(ts_result, 'Params size (MB): ')
    return total_params, trainable_params, non_trainable_params, params_size

def size_of_model(model):
    tmp_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
    np.save(tmp_name+'.npy', np.array(list(model.cpu().state_dict().items()), dtype=object), allow_pickle=True)
    pth_size = os.path.getsize(tmp_name+'.npy') * 8 * 0.001
    os.remove(tmp_name+'.npy')
    # returns size in kilo bits, getsize returns bytes
    return pth_size

def main(dataset, data_name, repeat_experiment, total_steps, config_name):

    if config_name == 'depth':
        configs = ModelConfig().get_depth_runs()
    elif config_name == 'tiny':
        configs = ModelConfig().get_tiny_runs()
    elif config_name == 'paper_comparison':
        configs = ModelConfig().get_paper_comparison_runs()
    elif config_name == 'paper_architecture':
        configs = ModelConfig().get_paper_architecture_runs()
    elif config_name == 'paper_siam_bonding':
        configs = ModelConfig().get_paper_siam_bonding_runs()
    else:
        raise ValueError('Cannot parse model config ', config_name)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    s = next(iter(dataloader))
    model_input = s['amplitude']
    SPLIT_SIZE = 441000//2 # 10 sec snippets

    REPEAT_EXPERIMENT = repeat_experiment # how many samples to use for averaging

    save_audio = [50,100,150,250,500,1000,2500,5000,10000]
    save_audio = [i for i in save_audio if i <= total_steps]

    RESULT_FOLDER_NAME = f'{data_name}_{config_name}_{repeat_experiment}samples_{total_steps}it'
    if not os.path.exists('results'):
        os.makedirs('results')
    folder_path = os.path.join('results', RESULT_FOLDER_NAME)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # store configs as json
    with open(os.path.join(folder_path, 'configs.txt'), 'w') as f:
        for i in range(len(configs)):
            f.write(str(dict(configs[i])))
            f.write('\n')

    audio_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs), len(save_audio), SPLIT_SIZE, 2))
    quant_audio_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs), len(save_audio), SPLIT_SIZE, 2))
    size_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs), len(save_audio)))
    quant_size_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs), len(save_audio)))
    orig_audio = np.zeros((REPEAT_EXPERIMENT, SPLIT_SIZE))
    time_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs), len(save_audio)))
    model_breakdown_matrix = np.zeros((REPEAT_EXPERIMENT, len(configs),4))

    for i, sample in tqdm(enumerate(dataloader), total=REPEAT_EXPERIMENT):
        if i == REPEAT_EXPERIMENT:
            break
            
        ground_truth = sample['amplitude'].cuda()
        model_input = sample['timepoints'].cuda()
            
        orig_audio[i, :] = ground_truth.squeeze().detach().cpu().numpy()
        
        ground_truth = ground_truth.repeat(1, 1, 2)
            
        for sweep in range(len(configs)):
            hidden_features = configs[sweep]['hidden_features']
            siam_features = configs[sweep]['siam_features']
            num_frq = configs[sweep]['num_frq']
            first_omega_0 = configs[sweep]['first_omega_0']
            hidden_omega_0 = configs[sweep]['hidden_omega_0']
            optimizer = configs[sweep]['optim']
            weight_decay = configs[sweep]['weight_decay']
            loss_fn = configs[sweep]['loss_fn']

            audio_siren = SiameseSiren(in_features=1, out_features=1, hidden_features=hidden_features, 
                                    siam_features=siam_features, first_omega_0=first_omega_0, 
                                    hidden_omega_0=hidden_omega_0, outermost_linear=True, num_frq=num_frq)

            audio_siren.cuda()
            optim = optimizer(lr=1e-4, params=audio_siren.parameters())
            if weight_decay is not None:
                optim = optimizer(lr=1e-4, params=audio_siren.parameters(), weight_decay=weight_decay)

            best_loss = float('inf')
            best_model = audio_siren.state_dict()
            
            elapsed_time = 0
            for step in range(total_steps):
                st = time.time()
                
                model_output, coords = audio_siren(model_input)
                
                loss = loss_fn(model_output, ground_truth)
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                elapsed_time += (time.time() - st)
                
                if loss < best_loss:
                    best_loss = loss
                    best_model = audio_siren.state_dict()

                if step+1 in save_audio:
                    siren = SiameseSiren(in_features=1, out_features=1, hidden_features=hidden_features, 
                                      siam_features=siam_features, first_omega_0=first_omega_0, 
                                      hidden_omega_0=hidden_omega_0, outermost_linear=True, num_frq=num_frq)
                    siren.load_state_dict(best_model)
                    size_matrix[i, sweep, save_audio.index(step+1)] = size_of_model(siren)
                    audio_matrix[i, sweep, save_audio.index(step+1), :, :] = siren(model_input.cpu())[0].detach().cpu().numpy().squeeze()
                    siren_qint8 = torch.quantization.quantize_dynamic(siren, {torch.nn.Linear}, dtype=torch.qint8)
                    quant_audio_matrix[i, sweep, save_audio.index(step+1), :, :] = siren_qint8(model_input.cpu())[0].detach().cpu().numpy().squeeze()
                    quant_size_matrix[i, sweep, save_audio.index(step+1)] = size_of_model(siren_qint8)
                    
                    time_matrix[i, sweep, save_audio.index(step+1)] = elapsed_time
    
            total_params, trainable_params, non_trainable_params, params_size = get_model_parameters(summary_string(siren.cuda(), input_size=(1, 220500, 1))[0])
            model_breakdown_matrix[i, sweep, :] = [total_params, trainable_params, non_trainable_params, params_size]
        
    # store results in npy file
    np.save(os.path.join(folder_path, f"{data_name}_audio_{save_audio}_{REPEAT_EXPERIMENT}samples.npy"), audio_matrix)
    np.save(os.path.join(folder_path, f"{data_name}_quant_audio_{save_audio}_{REPEAT_EXPERIMENT}samples.npy"), quant_audio_matrix)
    np.save(os.path.join(folder_path, f"{data_name}_model_size_{total_steps}steps_{REPEAT_EXPERIMENT}samples.npy"), size_matrix)
    np.save(os.path.join(folder_path, f"{data_name}_quant_model_size_{total_steps}steps_{REPEAT_EXPERIMENT}samples.npy"), quant_size_matrix)
    np.save(os.path.join(folder_path, f"{data_name}_orig_audio_{REPEAT_EXPERIMENT}samples.npy"), orig_audio)
    np.save(os.path.join(folder_path, f"{data_name}_time_{save_audio}steps_{REPEAT_EXPERIMENT}samples.npy"), time_matrix)
    np.save(os.path.join(folder_path, f"{data_name}_model_breakdown_{total_steps}steps_{REPEAT_EXPERIMENT}samples.npy"), model_breakdown_matrix)
    
    # shutil.make_archive(folder_path, 'zip', root_dir='./results', base_dir=RESULT_FOLDER_NAME)
    # shutil.rmtree(folder_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="gtzan or librispeech", type=str, default="gtzan")
    parser.add_argument("-r", "--repeat", help="number of times to repeat experiment", type=int, default=2)
    parser.add_argument("-t", "--total", help="total number of steps to train", type=int, default=250)
    parser.add_argument("-c", "--config", help="which model config to run", type=str, default='tiny')
    parser.add_argument("-f", "--folder", help="folder location of gtzan audio data", type=str, default='data')
    args = parser.parse_args()
    
    if args.dataset == 'gtzan':
        dataset = AudioFileDataset(f'{args.folder}/**/*.wav', start_time_sec=0, end_time_sec=10)
        name = 'gtzan'
    elif args.dataset == 'librispeech':
        dataset = LibriSpeechDataset(start_time_sec=0, end_time_sec=10)
        name = 'librispeech'
        
    print("Running on dataset:", name)    
    main(dataset, name, args.repeat, args.total, args.config)