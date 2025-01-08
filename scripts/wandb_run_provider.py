import wandb

class WanDBRunProvider(object):
    '''
    A class to provide a wandb run object to the experiment.
    Should use static method to get the run object.
    '''

    _run = None

    def __init__(self, *args, **kwargs):
        WanDBRunProvider._run = wandb.init(*args, **kwargs)

    @staticmethod
    def get_run():
        return WanDBRunProvider._run
    
    @staticmethod
    def close():
        WanDBRunProvider._run.finish()
        WanDBRunProvider._run = None


