#my test module No.1

i=3
#i
print(i,'i',i)

def  func1():
    a=5
    b=7
    return a,b

class Cl1:
    d1=22
    def __init__(self) -> None:
        pass

#    self.d2=10.5

    def func2():
        e=55
        e2='yo'     
        print(e,e2)
        print(e)
        print('ho')
        return 'done'

cinst=Cl1()
dinst=Cl1()
print("calling func1")
a,b=func1()
print(a,b)

f=52
print(f, 'f')
print(cinst.d1)
print(Cl1.d1)
print(dinst.d1)
cinst.d1=30
print(cinst.d1)
print(Cl1.d1)
print(dinst.d1)
print('calling func2 as Cl1')
d=Cl1.func2()
print(d)
cinst.func2()
#dinst.func2



