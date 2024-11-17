import requests

def query_llm(question, context):
    api_key = "AIzaSyDZjx_T00dkfiMowtG1qJATnmW7hXlVMqI"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    headers = {
        "Content-Type": "application/json"
    }

    prompt = f"Question: {question}, Context: {context}"

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    print(response)

    if response.status_code == 200:
        result = response.json()
        try:
            answer = result['candidates'][0]['content']['parts'][0]['text']
            return answer
        except KeyError:
            return "Error: Unable to parse the response. Please check the API structure."
    else:
        return f"Error: {response.status_code}, {response.text}"
