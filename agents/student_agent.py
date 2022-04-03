# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from world import*
#World, PLAYER_1_NAME, PLAYER_2_NAME
from Node import*

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        '''
        #store all the paths into this array
        listOfPath = []

        #generate all simulations
        while time + space resources are left:
            #returns a path
            Path = stimulChild(chess_board, my_pos, adv_pos)
            listOfPath.append(Path)

        #rturn an array of UCTs, number of each direct action 
        nodeUCT = findAllUCT(listOfPath)

        bestNode = findBestNode(nodeUCT)

        return bestNode.my_pos, bestNode.dir
        



    
    
    #1 simulation 
    def stimulChild(chess_board, my_pose, adv_pos):

        
        #a path that is found by 1 simulation counsisting of nodes, attributes updated along the way
        path = []


    #backpropagation during each simulation

    def backpropagation(path): #do i need to do this or I can just increment the value of those numbers along the way? prob determined on how to implmemnet stimulChild function
        
    #calculate all UCT from any direct action, return a list of UCTs(how to find this,
    #by appending all UCTs into an array an find the node associated with the highest value? might need to break ties
    #how to search the Node given their UCT score, do i need to put UCT as the node's attribute. even if I do this, how can I find the Node by having its UCT score?
        
    def findAllUCT(listOfPath):

    #find the best UCT
    def findBestUCT(listOfUCTs):#idk how to implement this pls help

    #find the best Node
    def findBestNode(bestUCT):#same as the last question

    
       
    #An array of paths generated by simulations, paths is a list of tuples where (node,  a list of 2 numbers(number of visited + number of success)).

    #for each simulation, backpropagate values of nodes(visited times, number of success)

    #calculate UCTs for the direct children and find the best one

    #expand the node by returning this child
'''
