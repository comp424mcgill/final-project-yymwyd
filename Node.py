import math

class Node:
        #initiate Node 
        def __init__(self, my_pos, dir):
                #initialize node number of visited
                self.num_visited = 0
                #initialize node number of success
                self.num_success = 0
                self.my_pos = my_pos
                self.dir = dir

        #getter for  nodes
        def get_visits(self):
                return self.num_visited

        def get_wins(self):
                return self.num_success

        #setter for nodes
        def set_visits(self, v):
                self.num_visited = v

        def set_wins(self,w):
                self.num_success=w
                
                
                
