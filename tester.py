#tester file
from node import Node


#unit test for backpropagation
#assume all those are successful
def updateNode(n, success):
    newVisit = n.get_visits() + 1
    n.set_visits(newVisit)
    if(success):
        newWin = n.get_wins() + 1
        n.set_wins(newWin)

def main():
    listOfPaths = []
    updatedList = []
    path1 = []
    path2 = []
    path3 = []
    
    n1 = Node((1,2),0)
    n2 = Node((3,4),1)
    n3 = Node((5,6),2)
    path1.append(n1)
    path1.append(n2)
    path1.append(n3)

    n4 = Node((0,1),3)
    n5 = Node((2,2),1)
    n6 = Node((7,0),2)
    path2.append(n4)
    path2.append(n5)
    path2.append(n6)

    n7 = Node((6,5),0)
    n8 = Node((1,1),1)
    n9 = Node((3,2),2)
    path3.append(n7)
    path3.append(n8)
    path3.append(n9)
    listOfPaths.append(path1)
    listOfPaths.append(path2)
    listOfPaths.append(path3)
    #show values before backpropagation
    for list in listOfPaths:
        for n in list:
            print("This is the original number of success:\n", n.get_wins())
            print("This is the original number of visited:\n", n.get_visits())


    updatedPath1 = backpropagation(path1, True)
    updatedPath2 = backpropagation(path2, False)
    updatedPath3 = backpropagation(path3, False)

    updatedList.append(updatedPath1)
    updatedList.append(updatedPath2)
    updatedList.append(updatedPath3)

    #show values are updated
    for list in updatedList:
        print("one path\n")
        for n in list:
            print("This is the updated number of success:\n", n.get_wins())
            print("This is the updated number of visited:\n", n.get_visits())

if __name__ == "__main__":
    main()
