# Wiki Learn (Flask Backend)

This is a companion app to the main [Wiki Learn](https://github.com/QED0711/wiki_learn) project. The code contained here is a Flask backend. See the main Wiki Learn project for more details on the models and purpose of this application. See [here](https://github.com/QED0711/wiki_learn_react) for the React frontend of this application.

If you would like to clone and run a version of this backend on your local device, see installation instructions below. 

___

## Installation

Once forked, clone this repository and cd into the project directory:

```
$ git clone git@github.com:QED0711/wiki_learn_flask.git

$ cd wiki_learn_flask

```

Once in the directory, you will need to install all necessary packages (optionally, you can set up a virtual environment for this).

The fastest way to install the packages is through the `requirements.txt` file. 

```
$ pip install -r requirements.txt
```

When all packages have finished loading, you can run a local backend server on port 5000 with the usual flask command: 

```
$ flask run
```

___

## Notes: 

If you get an error related to the `keys` file not being present, simply comment out or delete that import and associated code in the `server.py` file. The keys file holds private access keys for database reading/writing, which should not be necessary for basic functionality in a local copy. 

In order to interact with this backend, you will also need to install and run a local version of the [React frontend](https://github.com/QED0711/wiki_learn_react).