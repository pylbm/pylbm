import notebookgenerator

def init():
    print('Generating notebooks.')
    notebookgenerator.main()

def clean(app, *args):
    print('Remove rst notebook files.')
    #notebookgenerator.clean()

def setup(app):
    init()
    app.connect('build-finished', clean)
