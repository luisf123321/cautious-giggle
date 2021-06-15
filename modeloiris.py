#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:41:40 2018

@author: macbookpro
"""
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
@app.route('/')
def index():
    return 'Hello world'

@app.route('/clasifica', methods=['POST'])
def classify():
    file = open("knn.model", 'rb')
    knn = joblib.load(file)
    file.close()
    json_data = request.json
    _id = json_data['_id']
    id = json_data['id']
    sl = json_data['sl']
    sw = json_data['sw']
    pl = json_data['pl']
    pw = json_data['pw']
    #Iris-setosa
    
    datos = np.array([sl,sw,pl,pw], ndmin = 2)
    predictions = knn.predict(datos)
    #print(type(predictions))
    #print(predictions)
    post = {"_id": _id, "id": id, "sl": sl,"sw": sw, "pl":pl, "pw": pw , "clase": predictions[0] }
    print((post))
    
    return str(post)
    


if __name__ == '__main__':
    app.run()
