from espnet2.bin.tts_inference import Text2Speech
import numpy as np
import soundfile as sf
import torch

# inference from trained model
def tts_inference(train_config, model_file, text, save_pth, use_sids=False, sids=None):
    tts = Text2Speech.from_pretrained(train_config=train_config,model_file=model_file)
    
    if use_sids:
        sids = torch.tensor(sids)
        wav = tts(text, sids=sids)["wav"]
    else:
        wav = tts(text)["wav"]
        
    sf.write(save_pth + ".wav", wav.numpy(), tts.fs, "PCM_16")
    return wav

# concat wav
def concat_wav(wav1, wav2):
    return torch.cat([wav1, wav2])

def main():
    # config
    dir_path = "/mnt/aoni04/hsieh/newsTTS_project/role_newsTTS/exp/tts_global_condition_cgan_vits_raw_phn_jaconv_pyopenjtalk_prosody/"
    model_type = "train.total_count.ave_10best.pth"
    filename = "waseda_pcl_ucd_news_039B00"
    # text_test = "一緒に<e>ヨガの</e><e>ポーズを</e>披露したんだって"
    # text_emp = "一緒にヨガの<e>ポーズを</e>披露したんだって"
    # text_emp1 = "一緒に<e>ヨガの</e>ポーズを披露したんだって"
    # text_emp2 = "<e>一緒に</e>ヨガのポーズを披露したんだって"
    text = "キンドルアンリミテッドっていうサービスなんだけど"
    
    # inference
    tts_inference(dir_path+"config.yaml", dir_path+model_type, text, dir_path+filename, use_sids=True, sids=1)
if __name__ == "__main__":
    main()