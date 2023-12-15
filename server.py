import math
#import pyini
from flask import Flask, request, jsonify
import subprocess
import os
import glob
import json
import tempfile
import torch
import nemo.collections.asr as nemo_asr
import onnxruntime
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER
from extract_info import extract_audio, delete_files_in_directory
import numpy as np
import time

app = Flask(__name__)

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

ort_session = onnxruntime.InferenceSession('pretrained/qn_Lr_0001_bs_8_epoch_3_1.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def setup_transcribe_dataloader(cfg, vocabulary):
    config = {
        'manifest_filepath': os.path.join(cfg['temp_dir'], 'manifest.json'),
        'sample_rate': 16000,
        'labels': vocabulary,
        'batch_size': min(cfg['batch_size'], len(cfg['paths2audio_files'])),
        'trim_silence': True,
        'shuffle': False,
    }
    dataset = AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=None,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', True),
        parser=config.get('parser', 'en'),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )

def audio_to_text(files):
    #print(files)
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for audio_file in files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': files, 'batch_size': 4, 'temp_dir': tmpdir}
        all_text = []
        temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary)
        for test_batch in temporary_datalayer:
            #print(test_batch[0])
            processed_signal, processed_signal_len = quartznet.preprocessor(
                input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
            )
            #print(1)
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal),}
            ologits = ort_session.run(None, ort_inputs)
            alogits = np.asarray(ologits)
            logits = torch.from_numpy(alogits[0])
            greedy_predictions = logits.argmax(dim=-1, keepdim=False)
            wer = WER(decoding=quartznet.decoding, use_cer=False)
            hypotheses, _ = wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
            if isinstance(hypotheses, list):
                for text in hypotheses:
                    all_text.append(text)
            else:
                all_text.append(hypotheses)
        return all_text

@app.route('/extract_audio', methods=['POST'])
def extract():
    if "audio" not in request.files:
        return jsonify({'error': 'No audio file uploaded'})

    all_audio_file = request.files.getlist('audio')
    #print(all_audio_file)
    t1 = time.time()

    all_audio_info = []
    for audio_file in all_audio_file:
        count_ = len(glob.glob('audio_files\\*.wav'))
        #print(audio_file.filename)
        audio_path = "audio_files\\" + str(audio_file.filename).split('\\')[-1]
        #print(audio_path)
        audio_file.save(audio_path)

    all_audio_path = glob.glob('audio_files\\*.wav')
    all_text = audio_to_text(all_audio_path)
    #print(all_text)
    for i, audio_path in enumerate(all_audio_path):
        audio_info = extract_audio(audio_path)
        if 'error' not in audio_info:
            audio_info['text'] = all_text[i]
        #print(audio_info)
        all_audio_info.append(audio_info)

    #print(os.listdir('audio_files'))
    delete_files_in_directory('audio_files')
    t2 = time.time()
    print('Processing time: ', t2-t1)
    return jsonify(all_audio_info)

if __name__ == "__main__":
    app.run(debug=True)