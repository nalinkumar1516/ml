import requests
url = "http://127.0.0.1:8000/predict"

student_data = {
    "study_hours": 5,
    "attendance": 80,
    "previous_marks": 75,
}

response = requests.post(url, params=student_data)
print(response.json())


