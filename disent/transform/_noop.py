

class Noop(object):
    """
    Transform that does absolutely nothing!
    """

    def __call__(self, obs):
        return obs

    def __repr__(self):
        return f'{self.__class__.__name__}()'

