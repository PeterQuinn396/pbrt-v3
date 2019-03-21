# import numpy
# import torch
class testObject(object):

    def __init__(self):

        self.data = []

    def add(self, val):
        self.data.append(val)

    def __str__(self):
        return "Data: " + str(self.data)

cdef public int addTest(int a, int b):
    return a+b

cdef public void testPrint():
    print("testing")


cdef public object createTestObject():
    obj = testObject()
    return obj

cdef public void addData(object p, int val):
    p.add(val)

cdef public char* printCls(object p):
    return bytes(str(p), encoding = 'utf-8')


cdef api void testPrint_api():
    print("testing")


