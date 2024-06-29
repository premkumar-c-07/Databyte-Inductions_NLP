from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('savedmodel.pkl', 'rb'))
count_vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        msglist = [message]
        testemail = count_vectorizer.transform(msglist)
        prediction = model.predict(testemail.toarray())[0]
        if prediction.argmax() == 1:
            result=f'{message}  -   Spam'
        else:
            result=f'{message}  -   Not Spam'

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
