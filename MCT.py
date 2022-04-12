import math

class Node:
        #initiate MCT 
        def __init__(self, num_visited, num_success):
                #initialize node number of visited
                self.num_visited = 0
                #initialize node number of success
                self.num_success = 0
                self.children = dict()
                
                
                
