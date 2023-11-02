from flask import Flask, render_template, request
import uuid
import datetime
import os
from lanchain_help import return_answer, htmlerize


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('in.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    input_data = request.form['input_data']
    logid=str(uuid.uuid4().hex)
    txtanswer= return_answer(input_data, logid)
    txtanswer= htmlerize(txtanswer)
    return render_template('out.html', question= input_data, answer=txtanswer, uuidtxt=logid)

@app.route('/badbot', methods=['POST'])
def process_data_bad():
    badreason = request.form['badreason']
    uniqueid = request.form['uuidtxt']
    input_data = request.form['input_data']
    txtanwer = request.form['output_data']
    filename = os.path.join("home/AVAAbschluss/flask/logs", uniqueid +"_bad.txt")
    with open(filename, "w") as f:
        f.write("Suggestion: ")
        f.write(badreason)
    return render_template('thanks.html', question=input_data, answer=txtanwer, badreason=badreason)


app.run()
