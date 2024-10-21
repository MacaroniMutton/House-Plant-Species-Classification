import requests

url = 'http://127.0.0.1:5000/predict'
file_path = r'C:\Users\Manan Kher\Downloads\plant_aloe_vera.jpg'

# Open the image file
with open(file_path, 'rb') as img_file:
    files = {'file': img_file}
    response = requests.post(url, files=files)

# Print the response from the API
print(response.json())
