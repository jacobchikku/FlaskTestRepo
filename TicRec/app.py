import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from nltk.corpus import stopwords
import string
def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [ request.form.values()]
    prediction = model.predict(int_features)
    #print(prediction)
    
    res = np.char.find(prediction, "SAS Service Request")

    if (res==1):
        prediction='This is an issue related to your SAS Service. Please log a Call with the Relevent team for immediate resolution'
    
    res = np.char.find(prediction, "SAS Access")
    if (res==1):
        prediction='This is a SAS Access issue. Please log a Call with the Relevent team for immediate resolution'

    return render_template('index.html', prediction_text='{}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)