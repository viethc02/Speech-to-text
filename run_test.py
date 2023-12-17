import requests
import time
import glob
import concurrent.futures

def upload_and_measure_time(file_paths):
    requests_list = []
    file_list = []
    for file_path in file_paths:
        file_list.append(('audio', open(file_path, 'rb')))
    #print(file_list)
    start_time = time.time()
    request = requests.post('http://localhost:5000/extract_audio', files=file_list)
    if request.status_code == 200:
        print(f"File uploaded and processed successfully.")
    else:
        print(f"Error uploading file. Status code: {request.status_code}")    
    
    end_time = time.time()
    return request.text, end_time - start_time

file_path_10p = ['test_data\84-121123-0000.wav']
res, time_10p = upload_and_measure_time(file_path_10p)
print(res)
print(f"Time to upload and process 1 file: {time_10p} seconds")

file_paths_500 = glob.glob('test_data/*.wav')
#start_time_500 = time.time()
res, total_time = upload_and_measure_time(file_paths_500)

end_time_500 = time.time()
#total_time_500 = end_time_500 - start_time_500
print(res)
print(f"Time to upload and process 500 files: {total_time} seconds")
