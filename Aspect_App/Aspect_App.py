from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/index", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            #print(image)

            return render_template("index.html")
        else:
            pass #Error


    return render_template("index.html")

@app.route("/webcam", methods=["GET", "POST"])
def access_webcam():

    if request.method == "GET":

        return render_template("webcam.html")
    
if __name__ == "__main__":
    app.run(debug=True)