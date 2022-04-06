
class Node:
        #initiate Node 
        def __init__(self, my_pos, dir, parent):
                #initialize node number of visited
                self.num_visited = 0
                #initialize node number of success
                self.num_success = 0
                self.my_pos = my_pos
                self.dir = dir
                self.parent = parent
                self.children = dict() #(action:UCT)

        #getter for  nodes
        def get_visits(self):
                return self.num_visited

        def get_wins(self):
                return self.num_success

        def get_children(self):
                return self.children

        def get_pos(self):
                return self.my_pos

        def get_dir(self):
                return self.dir

        def get_parent(self):
                return self.parent

        #setter for nodes
        def set_visits(self, v):
                self.num_visited = v

        def set_wins(self,w):
                self.num_success=w
            
        def set_parent(self, p):
                self.parent = p
            
        #add children
        def set_children(self, c):
                self.children = c



                
                
