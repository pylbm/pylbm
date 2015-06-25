import notebookgenerator

def init():
    print('Generating notebooks.')
    notebookgenerator.main()

def clean(app, *args):
    notebookgenerator.clean()

def setup(app):
    init()
    app.connect('build-finished', clean)
