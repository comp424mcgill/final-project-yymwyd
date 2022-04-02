#tester file
from node import Node


#unit test for backpropagation
#assume all those are successful
def backpropagation(listOfPaths):
    pass
def main():
    listOfPaths = []
    Path1 = []
    Path2 = []
    Path3 = []
    
    N1 = Node((1,2),0)
    N2 = Node((3,4),1)
    N3 = Node((5,6),2)
    Path1.append(N1)
    Path1.append(N2)
    Path1.append(N3)

    N4 = Node((0,1),3)
    N5 = Node((2,2),1)
    N6 = Node((7,0),2)
    Path2.append(N4)
    Path2.append(N5)
    Path2.append(N6)

    N7 = Node((6,5),0)
    N8 = Node((1,1),1)
    N9 = Node((3,2),2)
    Path3.append(N7)
    Path3.append(N8)
    Path3.append(N9)

    listOfPaths.append(Path1)
    listOfPaths.append(Path2)
    listOfPaths.append(Path3)

    updatedList = backpropagation(listOfPaths)

    #show values before backpropagation
    for list in listOfPaths:
        for n in list:
            print("This is the updated number of success:\n", n.get_wins())
            print("This is the updated number of visited:\n", n.get_visits())

    '''
    NumPath = len(updatedList)
    #show values are updated
    for list in updatedList:
        for n in list:
            print("This is the updated number of success:\n", n.get_wins())
            print("This is the updated number of visited:\n", n.get_visits())
'''
if __name__ == "__main__":
    main()
