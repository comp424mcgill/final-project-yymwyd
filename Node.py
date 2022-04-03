class Node:
        #initiate Node 
        def __init__(self, my_pos, dir, parent = None):
                #initialize node number of visited
                self.num_visited = 0
                #initialize node number of success
                self.num_success = 0
                self.my_pos = my_pos
                self.dir = dir
                self.parent = parent
                self.children = dict #(action:visited time)

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
            
        def set_parent(self, p):
                self.parent = p
            
        #add children
        def add_child(self, c):
                self.children.append(c)

                
                
