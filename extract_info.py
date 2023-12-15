import os
import glob
import json
import tempfile
from flask import Flask, request, jsonify
import subprocess

def extract_audio(audio_path):
    try:
        # get infomation of file
        #print(audio_path)
        result = subprocess.run(['sox', '--i', audio_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

    except Exception as e:
        return jsonify({'error': 'Error get information of file'})

    sample_rate = None
    duration = None

    for line in output.split('\n'):
        if 'Sample Rate' in line:
            sample_rate = int(line.split(':')[1].strip())
        elif 'Duration' in line:
            #print(line)
            hours = int(line.split(':')[1].strip())
            minutes = int(line.split(':')[2].strip())
            seconds = round(float((line.split(':')[3].strip()).split(" ")[0]))
            duration = hours * 3600 + minutes * 60 + seconds
            # duration = int(line.split(':')[1].strip())

    if duration is not None:
        return {'name': audio_path[12:], 'duration': duration}
    else:
        return {'error': 'Unable to retrieve audio information'}

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")