class MyClass:
    var1 = [1, 1]

    def update(value):
        MyClass.var1[0] += value

    def __init__(self, value):
        self.value = value
        MyClass.update(value)

    def tester(self, value):
        print(MyClass.var1)
        MyClass.var1[0] += value
        print(MyClass.var1)


b = MyClass(1)
a = b
t1 = a.tester
print(type(t1))
t1(100)
t1 = a.tester
t1(100)
print(b.var1)
