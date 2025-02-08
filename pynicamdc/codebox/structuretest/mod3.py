class Cl1:
    import sys
    print(sys.path)
#    import mod4 as m4
#    o=m4.Office54()
#    o.recommend()
    from mod4 import Office54 as O54
    o=O54()
    o.recommend()

    d1=22
#    def __init__(self) -> None:
#        pass

    def func2(self):
        e=55
        e2='yo'     
        print(e,e2)

cinst=Cl1()

print(cinst.d1)
cinst.func2()
#Cl1.func2(Cl1)
