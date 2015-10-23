from __future__ import print_function
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

class itemproperty(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(itemproperty, self).__init__(fget, fset, fdel, doc)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj)

class item2property(property):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(item2property, self).__init__(fget, fset, fdel, doc)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj, True)

class bounditemproperty(property):
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

class bound2itemproperty(property):
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

if __name__ == '__main__':
    import numpy as np
    class test(object):
        def __init__(self):
            self._m = np.arange(10).reshape((2, 5))
            self.nv_ptr = [0, 3, 8]
            self._m2 = np.arange(5*self.nv_ptr[-1]).reshape((self.nv_ptr[-1], 5))

        @itemproperty
        def m(self, i):
            return self._m[i]

        @m.setter
        def m(self, i , value):
            self._m[i] = value

        @item2property
        def m2(self, i, j):
            return self._m2[self.nv_ptr[i] + j]

        @m2.setter
        def m2(self, i, j, value):
            self._m2[self.nv_ptr[i] + j] = value


    a = test()
    print(a.m[1])
    print(a.m[:])
    a.m[1] = 1.
    print(a.m2[0][1])
    a.m2[0][1] = 1.
    print(a.m2[0][1], a.m2[0][1][1:])
    a.m2[0][1][1:] = 2
    print(a._m2)
