


class A():
    def __init__(self,val, CL):
        self.CL = CL
        self.val = val

    def a(self):
        return self.val

    def c(self):
        return self.CL(-self.val)


class B(A):
    def __init__(self, val):
        super().__init__(val, B)

    def a(self):
        return self.val**3

    
bb = B(3)
print(bb.a())    # 27

cc = bb.c()
print(cc.a())    # -27


    
