from espnet2.bin.tts_inference import Text2Speech
import numpy as np
import soundfile as sf
import torch

# inference from trained model
def inference(train_config, model_file, text, save_pth):
    tts = Text2Speech.from_pretrained(train_config=train_config,model_file=model_file)
    wav = tts(text)["wav"]
    sf.write(save_pth + ".wav", wav.numpy(), tts.fs, "PCM_16")
    return wav

# concat wav
def concat_wav(wav1, wav2):
    return torch.cat([wav1, wav2])

def main():
    # config
    model_pth = "/home/hsieh/espnet/egs2/nhk_radio/tts1/exp/tts_train_vits_44k_raw_phn_jaconv_pyopenjtalk_prosody/train.total_count.ave_10best.pth"
    train_config_pth = "/home/hsieh/espnet/egs2/nhk_radio/tts1/exp/tts_train_vits_44k_raw_phn_jaconv_pyopenjtalk_prosody/config.yaml"
    save_pth = "/home/hsieh/espnet/egs2/nhk_radio/tts1/test"
    text = "首都テヘランでは22日大規模な葬儀が行われています"
    text_test = "国立公園局は男性が単独で登山中16日に急な斜面でかつらくしたとみられるとしています"

    # inference
    inference(train_config_pth, model_pth, text_test, save_pth)

if __name__ == "__main__":
    main()