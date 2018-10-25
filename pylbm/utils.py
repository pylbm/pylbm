# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

#pylint: disable=invalid-name

class itemproperty(property):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj)

class item2property(property):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj, True)

class bounditemproperty:
    def __init__(self, item_property, instance, nextItem=False):
        self.__item_property = item_property
        self.__instance = instance
        self.nextItem = nextItem

    def __getitem__(self, key):
        fget = self.__item_property.fget
        if fget is None:
            raise AttributeError("unreadable attribute item")
        if self.nextItem:
            return bound2itemproperty(self.__item_property, self.__instance, key)
        else:
            return fget(self.__instance, key)

    def __setitem__(self, key, value):
        fset = self.__item_property.fset
        if fset is None:
            raise AttributeError("can't set attribute item")
        fset(self.__instance, key, value)

class bound2itemproperty:
    def __init__(self, item_property, instance, key):
        self.__item_property = item_property
        self.__instance = instance
        self.key = key

    def __getitem__(self, key):
        fget = self.__item_property.fget
        if fget is None:
            raise AttributeError("unreadable attribute item")
        return fget(self.__instance, self.key, key)

    def __setitem__(self, key, value):
        fset = self.__item_property.fset
        if fset is None:
            raise AttributeError("can't set attribute item")
        fset(self.__instance, self.key, key, value)

def header_string(title):
    from colorama import init, Fore, Style, Back
    init()
    bar = '+' + '-'*(len(title)+2) + '+'
    output = '\n| %s |\n'%title
    return Fore.BLUE + bar + output + bar + Fore.RESET