import os
import glob
import subprocess
import tarfile
import wget
import librosa
import json



data_dir = './data/'

if not os.path.exists(data_dir):
  os.makedirs(data_dir)


print("******")
if not os.path.exists(data_dir + '/dev-clean.tar.gz'):
    dev_clean_url = 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
    dev_clean_path = wget.download(dev_clean_url, data_dir)
    print(f"Dataset downloaded at: {dev_clean_path}")
else:
    print("Tarfile already exists.")
    dev_clean_path = data_dir + '/dev-clean.tar.gz'

if not os.path.exists(data_dir + '/LibriSpeech/dev-clean/'):
    tar = tarfile.open(dev_clean_path)
    tar.extractall(path=data_dir)

print("Converting .flac to .wav...")

flac_list = glob.glob(data_dir + '/LibriSpeech/dev-clean/**/**/*.flac', recursive=True)
for flac_path in flac_list:
    wav_path = flac_path[:-5] + '.wav'
    cmd = ['sox', flac_path, wav_path]
    subprocess.run(cmd, shell=True)
print("Finished conversion.\n******")

print("******")
if not os.path.exists(data_dir + '/test-clean.tar.gz'):
    test_clean_url = 'https://www.openslr.org/resources/12/test-clean.tar.gz'
    test_clean_path = wget.download(test_clean_url, data_dir)
    print(f"Dataset downloaded at: {test_clean_path}")
else:
    print("Tarfile already exists.")
    test_clean_path = data_dir + '/test-clean.tar.gz'

if not os.path.exists(data_dir + '/LibriSpeech/test-clean/'):
    # Untar and convert .sph to .wav (using sox)
    tar = tarfile.open(test_clean_path)
    tar.extractall(path=data_dir)

print("Converting .flac to .wav...")

flac_list = glob.glob('./LibriSpeech/test-clean/**/**/*.flac', recursive=True)
for flac_path in flac_list:
    wav_path = flac_path[:-5] + '.wav'
    cmd = ["sox", flac_path, wav_path]
    subprocess.run(cmd, shell=True)
print("Finished conversion.\n******")

def build_manifest(transcripts_path, manifest_path, wav_path):
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'a') as fout:
            for line in fin:
                file_id = line[: line.find(' ')]
                transcript = line[line.find(' ')+1 : -1].lower()

                audio_path = os.path.join(wav_path + '/', file_id + '.wav')

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')

print("******\n build dev manifest")
dir_path = data_dir + '/LibriSpeech/dev-clean'
train_manifest = data_dir + '/dev_manifest.json'
list_dir = os.listdir(dir_path)
if not os.path.isfile(train_manifest):
  print('transcripting')
  for file_root in list_dir:
    dir_root_path = os.path.join(dir_path + '/', file_root)
    list_dir_root = os.listdir(dir_root_path)
    for file_children in list_dir_root:
      dir_children_path = os.path.join(dir_path + '/', file_root + '/', file_children)
      transcripts_path = dir_children_path + '/' + file_root + '-' + file_children + '.trans.txt'
      build_manifest(transcripts_path, train_manifest, dir_children_path)

print("***Done***")

print("******\n build test manifest")
dir_path_test = data_dir + '/LibriSpeech/test-clean'
test_manifest = data_dir +'/test_manifest.json'
list_dir_test = os.listdir(dir_path_test)
if not os.path.isfile(test_manifest):
  print('transcripting')
  for file_root in list_dir_test:
    dir_root_path = os.path.join(dir_path_test + '/', file_root)
    list_dir_root = os.listdir(dir_root_path)
    for file_children in list_dir_root:
      dir_children_path = os.path.join(dir_path_test + '/', file_root + '/', file_children)
      transcripts_path = dir_children_path + '/' + file_root + '-' + file_children + '.trans.txt'
      build_manifest(transcripts_path, test_manifest, dir_children_path)

print("***Done***")