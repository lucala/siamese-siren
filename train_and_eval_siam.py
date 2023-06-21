import os
import random
import string

import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchsummary
from model import SiameseSiren
from dataset import AudioFileDataset, LibriSpeechDataset
from torch.utils.data import DataLoader
from tqdm import trange

def plot_spectrogram(y, path):
    fig = plt.figure()
    S = librosa.feature.melspectrogram(y=y, sr=22050)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, y_axis='mel', sr=22050)
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',transparent=True, pad_inches=0)
    
def find_value_for_string(string, search_string):
    return string[string.find(search_string)+len(search_string):string.find('\n', string.find(search_string))]

def get_model_parameters(ts_result):
    total_params = find_value_for_string(ts_result, 'Total params: ')
    trainable_params = find_value_for_string(ts_result, 'Trainable params: ')
    non_trainable_params = find_value_for_string(ts_result, 'Non-trainable params: ')
    params_size = find_value_for_string(ts_result, 'Params size (MB): ')
    return total_params, trainable_params, non_trainable_params, params_size

def size_of_model(model):
    tmp_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
    np.save(tmp_name+'.npy', np.array(list(model.cpu().state_dict().items())), allow_pickle=True)
    pth_size = os.path.getsize(tmp_name+'.npy') * 8 * 0.001
    os.remove(tmp_name+'.npy')
    # returns size in kilo bits, getsize returns bytes
    return pth_size

def save_audio(y, path):
    torchaudio.save(path, y, 22050)

def export_to_onnx(torch_model, folder_path):
    # Input to the model
    x = torch.randn(1, 220500, 1, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model, x, f"{folder_path}/siam.onnx", export_params=True, opset_version=10, 
                    do_constant_folding=True, input_names = ['input'], output_names = ['output'], 
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

def run_training_and_eval(runs, val_timepoints, val_amplitude, total_steps, result_folder_name, audio_save_interval, img_save_interval, mel_spec_quant):
    
    result_folder_name_img = f'{result_folder_name}_imgs'
    folder_path = os.path.join('results', result_folder_name_img)
    folder_path_imgs = os.path.join('results', result_folder_name_img)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(folder_path_imgs):
        os.makedirs(folder_path_imgs)
    
    plot_spectrogram(val_amplitude.squeeze().detach().cpu().numpy(), 
                     f'{folder_path_imgs}/{result_folder_name}_ground_truth_mel_spectrogram.png')
    save_audio(val_amplitude.squeeze(-1).detach().cpu(), f"{folder_path}/{result_folder_name}_ground_truth.wav")

    for sweep in range(len(runs)):
        hidden_features = runs[sweep]['hidden_features']
        num_frq = runs[sweep]['num_frq']
        first_omega_0 = runs[sweep]['first_omega_0']
        hidden_omega_0 = runs[sweep]['hidden_omega_0']
        optimizer = runs[sweep]['optim']
        weight_decay = runs[sweep]['weight_decay']
        loss_fn = runs[sweep]['loss_fn']
        siam_features = runs[sweep]['siam_features']
        separate_last_layer = runs[sweep]['separate_last_layer']

        audio_siren = SiameseSiren(in_features=1, out_features=1, hidden_features=hidden_features, siam_features=siam_features,
                                    first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0, outermost_linear=True, 
                                    num_frq=num_frq, separate_last_layer=separate_last_layer)
        export_to_onnx(audio_siren, folder_path)
        audio_siren.cuda()
        torchsummary.summary(audio_siren, input_size=(1, 220500, 1))

        optim = optimizer(lr=1e-4, params=audio_siren.parameters())
        if weight_decay is not None:
            optim = optimizer(lr=1e-4, params=audio_siren.parameters(), weight_decay=weight_decay)

        best_loss = float('inf')
        audio_siren.cuda()
        ground_truth = val_amplitude.cuda()
        model_input = val_timepoints.cuda()
        tr = trange(total_steps, leave=True)
        for step in tr:
            
            model_output, coords = audio_siren(model_input)
            loss = loss_fn(model_output, ground_truth.repeat(1,1,2))

            audio_siren.cuda()

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if loss < best_loss:
                best_loss = loss
                torch.save(audio_siren.state_dict(), f"{folder_path}/{result_folder_name}_optimized_siren.pth")
                
            if step+1 in audio_save_interval:
                siren = SiameseSiren(in_features=1, out_features=1, hidden_features=hidden_features, siam_features=siam_features,
                                    first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0, outermost_linear=True, 
                                    num_frq=num_frq, separate_last_layer=separate_last_layer)
                siren.load_state_dict(torch.load(f"{folder_path}/{result_folder_name}_optimized_siren.pth"))
                siren_quant = torch.quantization.quantize_dynamic(siren, {torch.nn.Linear}, dtype=torch.qint8)
                siren.cuda()
                
                wave0 = siren(model_input)[0].cpu()[:,:,0]
                wave1 = siren(model_input)[0].cpu()[:,:,1]
                wave_stereo = siren(model_input)[0].cpu().squeeze()
                save_audio(wave0, f"{folder_path}/{result_folder_name}_{step+1}_best_ch0.wav")
                save_audio(wave1, f"{folder_path}/{result_folder_name}_{step+1}_best_ch1.wav")
                save_audio(wave_stereo, f"{folder_path}/{result_folder_name}_{step+1}_best_stereo.wav")
                
                quant_wav0 = siren_quant(model_input.cpu())[0].cpu()[:,:,0]
                quant_wav1 = siren_quant(model_input.cpu())[0].cpu()[:,:,1]
                quant_wav_stereo = siren_quant(model_input.cpu())[0].cpu().squeeze()
                save_audio(quant_wav0, f"{folder_path}/{result_folder_name}_{step+1}_quant_best_ch0.wav")
                save_audio(quant_wav1, f"{folder_path}/{result_folder_name}_{step+1}_quant_best_ch1.wav")
                save_audio(quant_wav_stereo, f"{folder_path}/{result_folder_name}_{step+1}_quant_best_stereo.wav")
                
                noise = wave0 - wave1
                signal = torch.tensor(nr.reduce_noise(y=wave0.detach().cpu().numpy().squeeze(), sr=22050, 
                                                    y_noise=noise.detach().cpu().numpy().squeeze(), stationary=True))
                save_audio(noise, f"{folder_path}/{result_folder_name}_{step+1}_best_noise.wav")
                save_audio(signal.unsqueeze(0), f"{folder_path}/{result_folder_name}_{step+1}_best_nr.wav")
                
                noise = quant_wav0 - quant_wav1
                denoised_quant = torch.tensor(nr.reduce_noise(y=quant_wav0.detach().cpu().numpy().squeeze(), sr=22050, 
                                                y_noise=noise.detach().cpu().numpy().squeeze(), stationary=True))
                save_audio(noise, f"{folder_path}/{result_folder_name}_{step+1}_quant_best_noise.wav")
                save_audio(denoised_quant.unsqueeze(0), f"{folder_path}/{result_folder_name}_{step+1}_quant_best_nr.wav")
                
                denoised_stationary_quant = torch.tensor(nr.reduce_noise(y=quant_wav0.detach().cpu().numpy().squeeze(), sr=22050, stationary=True))                
                denoised_nostationary_quant = torch.tensor(nr.reduce_noise(y=quant_wav0.detach().cpu().numpy().squeeze(), sr=22050))
                save_audio(denoised_stationary_quant.unsqueeze(0), 
                           f"{folder_path}/{result_folder_name}_{step+1}_quant_best_nr_stationary.wav")
                save_audio(denoised_nostationary_quant.unsqueeze(0), 
                           f"{folder_path}/{result_folder_name}_{step+1}_quant_best_nr_no_stationary.wav")
                
            if step+1 in img_save_interval:
                # TODO reusing siren_quant from save_audio, this works as long as save_audio is the same as save_img
                if mel_spec_quant:
                    wave = siren_quant(model_input.cpu())[0].detach().cpu().numpy()
                    _quant = "_quant"
                else:
                    wave = siren(model_input)[0].detach().cpu().numpy()
                    _quant = ""
                wave0 = wave[:,:,0].squeeze()
                wave1 = wave[:,:,1].squeeze()
                denoised = nr.reduce_noise(y=wave0, sr=22050, y_noise=wave0-wave1, stationary=True)
                
                plot_spectrogram(wave0-wave1, f'{folder_path_imgs}/{result_folder_name}_{step+1}_noise_estimate.png')
                plot_spectrogram(wave0, f'{folder_path_imgs}/{result_folder_name}_{step+1}{_quant}_best_melspec.png')
                plot_spectrogram(denoised, f'{folder_path_imgs}/nr_{result_folder_name}_{step+1}{_quant}_best_melspec.png')
                plot_spectrogram(denoised_stationary_quant.detach().cpu().numpy(), 
                                 f'{folder_path_imgs}/nr_{result_folder_name}_{step+1}{_quant}_best_melspec_stationary_no_estimate.png')
                plot_spectrogram(denoised_nostationary_quant.detach().cpu().numpy(), 
                                 f'{folder_path_imgs}/nr_{result_folder_name}_{step+1}{_quant}_best_melspec_no_stationary_no_estimate.png')
                model_input.cuda()
                
            tr.set_description(f'loss: {loss.item():.8f}')
                
        with open(os.path.join(folder_path, 'configs.txt'), 'w') as f:
            f.write("fp32 size "+str(size_of_model(audio_siren))+"\n")
            f.write("quant size "+str(size_of_model(siren_quant))+"\n")
            total_params, trainable_params, non_trainable_params, params_size = get_model_parameters(torchsummary.summary_string(siren, input_size=(1, 220500, 1))[0])
            f.write("total params "+str(total_params)+"\n")
            f.write("trainable params "+str(trainable_params)+"\n")
            f.write("non trainable params "+str(non_trainable_params)+"\n")
            f.write("params size (MB) "+str(params_size)+"\n")
            print('total params: ', total_params)
            print('trainable params: ', trainable_params)
            print('non trainable params: ', non_trainable_params)
            print('params size: ', params_size)
            

if __name__ == '__main__':
    
    runs = []
    runs.append(
        {
        'hidden_features': [256, 256],
        'siam_features': [128],
        'num_frq': 16,
        'first_omega_0': 100,
        'hidden_omega_0': 100,
        'optim': torch.optim.Adam,
        'weight_decay': 1e-5,
        'loss_fn': F.mse_loss,
        'separate_last_layer': False,
        },
    )

    # how many steps to train and how often to save the results
    total_steps = 100000
    audio_save_interval = [1000, 5000, 10000, 25000, 50000, 100000]
    img_save_interval = [1000, 5000, 10000, 25000, 50000, 100000]

    # store spectrogram for quantized model? affects only images, not audio
    mel_spec_quant = True
    qq = '_quant' if mel_spec_quant else ''

    audio_example = 'choice'
    audio_path = librosa.ex(audio_example)
    
    result_folder_name = f'{audio_example}_2x256_1x128siam_PE_melspec' + qq
    
    val_loader = DataLoader(AudioFileDataset(audio_path, start_time_sec=0, end_time_sec=10), batch_size=1, pin_memory=True, num_workers=0)
    val_sample = next(iter(val_loader))
    val_timepoints = val_sample['timepoints'].cuda()
    val_amplitude = val_sample['amplitude'].cuda()
    
    run_training_and_eval(runs, val_timepoints, val_amplitude, total_steps, result_folder_name, audio_save_interval, img_save_interval, mel_spec_quant)