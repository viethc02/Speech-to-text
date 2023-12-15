import requests
import time
import glob
import concurrent.futures

def upload_and_measure_time(file_path):
    start_time = time.time()

    # Tạo yêu cầu HTTP để upload file
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file)}
        response = requests.post('http://localhost:5000/extract_audio', files=files)

    end_time = time.time()

    # Xử lý phản hồi từ server
    if response.status_code == 200:
        print(f"File {file_path} uploaded and processed successfully.")
    else:
        print(f"Error uploading file {file_path}. Status code: {response.status_code}")

    # Trả về thời gian xử lý
    return end_time - start_time

file_path_10p = 'test_data/84-121123-0000.wav'
time_10p = upload_and_measure_time(file_path_10p)
print(f"Time to upload and process 1 file (10p): {time_10p} seconds")

file_paths_500 = glob.glob('test_data/*.wav')
start_time_500 = time.time()

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(upload_and_measure_time, file_paths_500))

end_time_500 = time.time()
total_time_500 = end_time_500 - start_time_500

print(f"Time to upload and process 500 files: {total_time_500} seconds")
