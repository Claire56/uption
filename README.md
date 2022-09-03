# UpTion
UpTion is a webapp that uses machine learning to classify a startup as fundable or not based on its characteristics.

### Classify a satrtup
Users provide information related to their startups <br> <br>

<a href="https://github.com/claire-Kimbugwe">
    <img alt="graphs" src="/static/graphs.gif" width="900" height="500">
    </a>

## Table of Contents
* [Overview](#Overview)
* [Tech Stack](#Tech-Stack)
* [Setup and installation](#Setup-and-installation)
* [Demo](#Demo)
* [Future Features](#Future-Features)

## Overview
There are so many interprenuers with great ideas but without the money needed to put turn those ideas into solutions that solve day to day problems. Uption is a fun web app that that is used to brigde the gap between venture companies and interprenuers. It used machine learning to classify a startup as safe for investimates or not and hopes to also match specific investors with startups i the future. 
#### Usage
once the user gets to my homepage, I presented with a form to present to provide information about their startup. This information is then used to classify the startup as either safe for funding or not
 

## Tech Stack
Data Wrangling: Pandas, Numpy , seaborn, matplotlib <br>
Framework: Flask <br>
Backend: Python, Tox,SciKit_Learn, pytest, <br>
Frontend: Javascript , AJAX, JSON , JQuery, Jinja, HTML, CSS, Bootstrap <br>
Model Monitoring: Prometheus, Grafana, <br>
Cloud services:

## Demo
### Data Wrangling and Exploration
I used jupyter notebook on anaconda to wrangle data and to explore trends and relations <br>
Below are a few visuals from my notebook 

### correlational Matrix

<a href="https://github.com/claire_kimbugwe">
    <img alt="explore" src="/static/explore1.gif" width="800">
    </a>

### Missing Values Table

<a href="https://github.com/claire_kimbugwe">
    <img alt="explore" src="/static/explore3.gif" width="800">
    </a>



### ML Algorithmns <br>
I had a chance to explore two different algorithmns, the logistic regression model and gradient Boasting classifier. I decided to use the Gradient boasting classifier model because of its high prediction score and low mean absolute error rate

<a href="https://github.com/claire_kimbugwe">
    <img alt="explore" src="/static/ML.gif" width="800">
    </a>

<br> <br>
### HOMEPAGE <br>
Below is my landing page <br><br>

<a href="https://github.com/claire_kimbugwe">
    <img alt="explore" src="/static/home.gif" width="800">
    </a>

#### Get information about user's startup <br>
On this page the users provide their startup information <br><br>

<a href="https://github.com/claire_kimbugwe">
    <img alt="explore" src="/static/features.gif" width="800">
    </a>

<br> <br>
### Provide result to the users 
Here the users recieve their results <br>

![graphs](/static/graphs.gif)
<br> <br>
## Setup and installation
On local machine, go to desired directory. Clone  repository:

$ git clone https://github.com/Claire56/uption.git <br>
Create a virtual environment in the directory:

$ virtualenv env<br>
Activate virtual environment:<br>

$ source env/bin/activate<br><br>
Install dependencies:<br>
$ pip install -r requirements.txt <br>
Run Tests:<br><br>

$ py.test<br>
More Information regarding docker coming soon:<br>

<!-- $ python3 -i model.py<br>
>>> db.create_all() <br>
Seed database:

$ python3 -i seed.py <br>
Run app:

$ python3 server.py <br>
Navigate to localhost:5000 in browser. -->



## Future Features
* utelise housing API's to get running data that will be used in training the machine learning model
* Add a login page for frequent visitors 



