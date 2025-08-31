import requests

def test():
    print("Starting test")
    try:
        resp = requests.post('http://127.0.0.1:8080/v1/eval/text', json={'text': 'This is a test.'})
        print(resp.status_code)
        print(resp.json())
    except Exception as e:
        print(f"Error: {e}")

test()
