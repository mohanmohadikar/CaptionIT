# Copyright @ 2020 ABCOM Information Systems Pvt. Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import uuid
import flask
import urllib
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
import pickle
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#model = load_model(os.path.join(BASE_DIR , 'ModelWebApp.hdf5'))
model = load_model(os.path.join(BASE_DIR , 'model.h5'))


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']


def predict(filename , model):
	img = load_img(filename , target_size = (32 , 32))
	img = img_to_array(img)
	img = img.reshape(1 , 32 ,32 ,3)

	img = img.astype('float32')
	img = img/255.0
	result = model.predict(img)

	dict_result = {}
	for i in range(10):
	    dict_result[result[0][i]] = classes[i]

	res = result[0]
	res.sort()
	res = res[::-1]
	prob = res[:3]
	
	prob_result = []
	class_result = []
	for i in range(3):
		prob_result.append((prob[i]*100).round(2))
		class_result.append(dict_result[prob[i]])

	return class_result , prob_result




@app.route('/')
def home():
		return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
	error = ''
	target_img = os.path.join(os.getcwd() , 'static/images')
	if request.method == 'POST':
		if(request.form):
			link = request.form.get('link')
			try :
				resource = urllib.request.urlopen(link)
				unique_filename = str(uuid.uuid4())
				filename = unique_filename+".jpg"
				img_path = os.path.join(target_img , filename)
				output = open(img_path , "wb")
				output.write(resource.read())
				output.close()
				img = filename

				res = pred(img_path)

				#class_result , prob_result = predict(img_path , model)

				predictions = {
					  "class1":res,
					    "class2":"class_result[1]",
					    "class3":"class_result[2]",
					    "prob1": "prob_result[0]",
					    "prob2": "prob_result[1]",
					    "prob3": "prob_result[2]",
				}

			except Exception as e : 
				print(str(e))
				error = 'This image from this site is not accesible or inappropriate input'

			if(len(error) == 0):
				return  render_template('success.html' , img  = img , predictions = predictions)
			else:
				return render_template('index.html' , error = error) 

			

		elif (request.files):
			file = request.files['file']
			if file and allowed_file(file.filename):
				file.save(os.path.join(target_img , file.filename))
				img_path = os.path.join(target_img , file.filename)
				img = file.filename
				res = pred(img_path)

			#	class_result , prob_result = predict(img_path , model)
			#really nigga.
				predictions = {
					  "class1":res,
					    "class2":"class_result[1]",
					    "class3":"class_result[2]",
					    "prob1": "prob_result[0]",
					    "prob2": "prob_result[1]",
					    "prob3": "prob_result[2]",
				}

			else:
				error = "Please upload images of jpg , jpeg and png extension only"

			if(len(error) == 0):
				return  render_template('success.html' , img  = img , predictions = predictions)
			else:
				return render_template('index.html' , error = error)

	else:
		return render_template('index.html')







def extract_features(filename, model):
    try:
        image = Image.open(filename)
            
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
 	for word, index in tokenizer.word_index.items():
    	 if index == integer:
        	return word
 	return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def pred(img_path):
	max_length = 32
	tokenizer = pickle.load(open("static/tokenizer.p","rb"))
#	model = load_model(os.path.join(BASE_DIR , 'model.h5'))
	#model = load_model('models/model_9.h5')
	xception_model = Xception(include_top=False, pooling="avg")

	photo = extract_features(img_path, xception_model)
	img = Image.open(img_path)

	description = generate_desc(model, tokenizer, photo, max_length)
	return description




if __name__ == "__main__":
	app.run(debug = True)


