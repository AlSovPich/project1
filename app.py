from flask import Flask, render_template, request
from service_functions.validate_url import validate_url
from rag.rag import RAGModule
import ai.gemini
from sentence_transformers import SentenceTransformer

app = Flask(__name__, template_folder="templates")

module = RAGModule()

@app.route('/', methods=['GET', 'POST'])
def index():
    global module
    url = ""
    question = ""
    answer = ""
    error_message = None

    if request.method == 'POST':
        url = request.form.get('doc_url')
        question = request.form.get('question')

        if not validate_url(url):
            error_message = "Invalid URL format. Please enter a valid URL."
            print(error_message)

        if error_message is None:
            module.extract_info_from_url(url)

            context = module.query_chromadb(question)
            
            answer = ai.gemini.query_llm(question, context)

    return render_template('index.html', url=url, question=question, answer=answer, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)