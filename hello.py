import os
from bottle import route, run
 
@route("/")
def hello_world():
        return "Hello World!"
 
@route("/hello")
def hello_world():
        return "hello"
 
run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))