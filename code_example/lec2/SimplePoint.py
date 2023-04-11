
# A Simple Point class (represents an x,y coordinate)
#
class Point:
    # __init__ function is the constructor
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    
    #move the point by x,y    
    def moveBy(self,x,y):
        self.x+=x
        self.y+=y

    #move the point to x,y
    def moveTo(self,x,y):
        self.x=x
        self.y=y
     
    #calculate Hamming distance between two points   
    def distanceTo(self, p2):
        return abs(self.x-p2.x) + abs(self.y-p2.y)
        
    # __str__ generates string representation of objects   
    def __str__(self):
        return '['+str(self.x)+','+str(self.y)+']'


#p1=Point()
#print(p1)

#p2=Point(3,5)
#print(p2)

#p3=Point(10,30)
#print(p3)
