import os
import numpy as np
import wave
import glob
import random
from scipy.io import wavfile


def Write_Vox1_Mix_Config (input_dir, output_dir, trial, overlap_min, overlap_max, snr_min, snr_max):
    print("Read all paths in Vox1 trials")
    all_path = glob.glob(os.path.join(input_dir,'*/*/*.wav'))

    with open(trial, 'r') as f:
            print("Trials opened successfully")
            same_utt_pointer = f.tell() #pointer -> the beginning of this trial file
            buffer_utt = f.readline().strip().split()[2] #先定为trial第一行的test_utt信息
            segs_list = []
            enroll_spk_list = []
            count = 0
            f.seek(0)
            for line in f.readlines():
                segs = line.strip().split() #segs = [target/nontarget,enroll,test,trial_name]
                test_utt = segs[2]
                if test_utt == buffer_utt:
                    #说明还是同一组音频
                    count = count+1
                    segs_list.append(segs)
                    enroll_spk_list.append(segs[1].split('/')[0])
                else:
                    #说明读到了新的一组音频,因此要对旧的开始处理
                    print("last utt: {} owns {} lines, new utt: {}".format(buffer_utt, count, test_utt))
                    audio1_path, audio2_path, audio1_name, audio2_name = choose_utt(segs_list, enroll_spk_list, all_path)
                    snr, overlap_ratio = choose_param(audio1_path, audio2_path, overlap_min, overlap_max, snr_min, snr_max)
                    lookback(f,same_utt_pointer,snr,overlap_ratio,segs_list,audio2_name) #这一步开始写txt和写trial文件
                    #mix_audio = mixture(audio1_path, audio2_path, snr, overlap_ratio) #用于根据txt生成混合音频
                    #mix_audio = augment_wav.add_noise(mix_audio, audio_sr)
                    
                    #Re-count
                    same_utt_pointer = f.tell()
                    buffer_utt = test_utt
                    count = 1
                    segs_list = []
                    enroll_spk_list = []
                    segs_list.append(segs)
                    enroll_spk_list.append(segs[1].split('/')[0])

            audio1_path, audio2_path, audio1_name, audio2_name = choose_utt(segs_list, enroll_spk_list, all_path)
            snr, overlap_ratio = choose_param(audio1_path, audio2_path, overlap_min, overlap_max, snr_min, snr_max)
            lookback(f,same_utt_pointer,snr,overlap_ratio,segs_list,audio2_name) #这一步开始写txt和写trial文件


def lookback (f,same_utt_pointer,snr,overlap_ratio,segs_list,audio2_name):
    f.seek(same_utt_pointer) #回到同样utt的这一组的第一行音频
    for segs in segs_list: #写count次
        write_txt(segs, snr, overlap_ratio, audio2_name)
        f.seek(1+same_utt_pointer) #当前位置往后挪动一个



def choose_utt (segs_list, enroll_spk_list, all_path): 
    audio1_path = os.path.join(input_dir,segs_list[-1][2]) #读取test_audio的地址
    audio2_path = random.choice(all_path)
    while audio2_path.split('/')[-3] in enroll_spk_list or audio2_path.split('/')[-3] == audio1_path.split('/')[-3]:
        audio2_path = random.choice(all_path)

    audio1_name = audio1_path.replace(input_dir+'/', '')
    audio2_name = audio2_path.replace(input_dir+'/', '')
    return audio1_path, audio2_path, audio1_name, audio2_name



def choose_param (audio1_path, audio2_path, overlap_min, overlap_max, snr_min, snr_max):
    snr = random.randint(snr_min*2, snr_max*2)/2
    sr, audio1 = wavfile.read(audio1_path) 
    sr, audio2 = wavfile.read(audio2_path) 
    overlap_max = min(overlap_max, min(audio1.size, audio2.size)/(audio1.size + audio2.size - min(audio1.size, audio2.size)))
    overlap_ratio = random.uniform(overlap_min, overlap_max) #此时为overlap_length / union_mixed_length
    return snr, overlap_ratio



def mixture (audio1_path, audio2_path, snr, overlap_ratio):
    audio1 = wave.open(audio1_path,"rb") 
    audio2 = wave.open(audio2_path,"rb") 
    audio1_db = 10 * np.log10(np.mean(audio1 ** 2) + 1e-4)
    audio2_db = 10 * np.log10(np.mean(audio2 ** 2) + 1e-4)
    audio_length = audio1.size*(1-overlap_ratio) + audio2.size
    mix_audio = np.zeros(audio_length)
    mix_audio[:audio1.size] = audio1
    audio2_startpos = audio_length-audio2.size
    mix_audio[audio2_startpos:] = mix_audio[audio2_startpos:] + np.sqrt(10 ** ((audio1_db - audio2_db - snr) / 10)) * audio2_db
    return mix_audio


def save_waveform(output_dir, mix_audio, buffer_utt, audio1_name):
    path = os.path.join(output_dir,audio1_name.replace('.wav','-mix.wav'))
    folder = os.path.dirname(path)
    if not os.path.exists(folder):                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder)

    audio_file = wave.Wave_write(path)
    audio_file.setnchannels(1) # mono
    audio_file.setsampwidth(2)
    sampleRate = 16000.0 # hertz, for vox1
    audio_file.setframerate(sampleRate)
    audio_file.writeframes(array.array('h', mix_audio.astype(np.int16)).tobytes() )
    audio_file.close()


def write_txt (segs, snr, overlap_ratio, audio2_name):
    mix_file_name = 'mix_configure-'+segs[3]
    tral_file_name = 'trial-'+segs[3]
    txt_path = os.path.join(output_dir,mix_file_name)
    trial_all_path = os.path.join(output_dir,tral_file_name)
    with open(txt_path,"a") as f:
        f.write('place1 place2 '+ segs[0]+' '+segs[1] + ' '+segs[2]+' '+' mixed with '+audio2_name+' '+str(snr)+' '+str(overlap_ratio))
        f.write('\r\n')

    with open(trial_all_path,"a") as t:
        t.write(segs[0]+' '+segs[1]+' '+segs[2].replace('.wav','-mix.wav'))
        t.write('\r\n')


if __name__ == '__main__':
    input_dir = '/mnt/lustre/sjtu/shared/data/raa/oxford/voxceleb1_wav_v2'
    output_dir = '/mnt/lustre/sjtu/home/lyz19/remote/Target-Speaker-Recognition/eval_data/two_audio_aug'
    trial = '/mnt/lustre/sjtu/home/lyz19/remote/Target-Speaker-Recognition/eval_data/vox1_all_clean_sort'
    overlap_min = 0.0
    overlap_max = 0.5
    snr_min = -3.0
    snr_max = 3.0 
    random.seed(10) 
    Write_Vox1_Mix_Config (input_dir, output_dir, trial, overlap_min, overlap_max, snr_min, snr_max)