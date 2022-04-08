# Student agent: Add your own agent here
from agents.agent import Agent
import numpy as np
from Node import *
# from agent import Agent
import math
from time import sleep, time
from copy import deepcopy
import copy
import sys
from store import register_agent

import signal
from contextlib import contextmanager


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "testAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.cur_step = 0

    @contextmanager
    def time_limit(self,seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Execution timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.cur_step = self.cur_step + 1
        #if(self.cur_step == 1):
            #time_limit = time.time() + 29.5
        #else:
            #time_limit = time.time() + 1.5
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
        #print("adv_pos is", adv_pos)
        #print("my_pos is", my_pos)
        #print("original chess_board", chess_board)
        root = Node(adv_pos, None, None)
        root.set_level(1)
        endNode = None
        bestNode = None
        tempch = deepcopy(chess_board)
        original_c = deepcopy(chess_board)
        tree_self_turn = 1
        # adv pos
        p0_pos = adv_pos
        # my pos
        p1_pos = my_pos
<<<<<<< HEAD
<<<<<<< HEAD
        d = 1
        if(self.cur_step == 1):
            time_limit = time() + 29.5
        else:
            time_limit = time() + 1.5
        start_time = time()
        while True:
            if time() - start_time >= 1 or d > 3:
                #print("time difference",time() - start_time)
                #print("the loo[p is out1!!!!!!!")
                break
=======
        #d = 1

        with self.time_limit(2):
>>>>>>> parent of 542ad17 (time module)
=======
        #d = 1

        with self.time_limit(2):
>>>>>>> parent of 542ad17 (time module)
            #print("d is:", d)
            # if the node is a leaf node,
            if len(root.get_children()) == 0:
                # print("root don't have children")
                # if the node is never visited
                if root.get_visits() == 0:
                    # print("root is not visited")
                    # simulate and back propagate
                    # when adv node is the one need to be simulated(aka start from me)
                    if tree_self_turn == 1:
                        #print("level is odd, cur node is adv, next move is mine")
                        if root.get_parent() is not None:
                            p1_pos = root.get_parent().get_pos()
                        else:
                            p1_pos = my_pos
                        p0_pos = root.get_pos()
                    else:
                        #print("level is even, cur node is me, next move is adv")
                        p1_pos = root.get_pos()
                        #print("my pos is", p1_pos)
                        p0_pos = root.get_parent().get_pos()
                        #print("adv pos is", p0_pos)
                    #print("the node being simulated", root.get_pos(), root.get_dir())
                    totalNum, numsuccess = self.m_simulate(original_c, p1_pos, p0_pos, tree_self_turn)
                    #print("total num is:", totalNum, "num success is:", numsuccess)
                    # now the node is the real upper root
                    root = self.backpropagation(root, totalNum, numsuccess)
                    #print("after bp", root.get_pos(), root.get_dir())
                    #print("root num visited is,", root.get_visits(), "num wins is", root.get_wins())
                    while root.get_parent() is not None:
                        root = root.get_parent()
                # if the node is visited, but has no children, expand it
                else:
                    #print("the node to be expanded", root.get_pos(), root.get_dir())
                    if tree_self_turn == 1:
                        #print("level is odd, cur node is adv, next move is mine")
                        if root.get_parent() is not None:
                            #print("the third level should go in this block")
                            p1_pos = root.get_parent().get_pos()
                        else:
                            p1_pos = my_pos
                        p0_pos = root.get_pos()
                        self.expand(root, p1_pos, p0_pos, max_step, original_c)
                    else:
                        #print("level is even, cur node is me, next move is adv")
                        p1_pos = root.get_pos()
                        #print("my pos is", p1_pos)
                        p0_pos = root.get_parent().get_pos()
                        #print("adv pos is", p0_pos)
                        self.expand(root, p0_pos, p1_pos, max_step, original_c)

                    # if can't find more children return the leafNode
                    # if endNode is not None:
                    # break

            # if the node has children, find the leafNode which does not have children
            else:
                original_c = deepcopy(tempch)
                root = self.select(root, original_c)
                #print("the leaf node selected", root.get_pos(), root.get_dir())
                #print("the leaf node level is", root.get_level())
                if (root.get_level() % 2 == 1):
                    tree_self_turn = 1
                else:
                    tree_self_turn = 0
                #d = root.get_level()
                # print("updated chess board by selection", original_c)
<<<<<<< HEAD
<<<<<<< HEAD
        #print("leaf root is", root.get_pos(), root.get_dir())

        while root.parent is not None:
            root = root.parent
        #print("root is", root.get_pos(), root.get_dir())
        bestNode = self.findBestUCT(root)
=======

        while root.parent is not None:
            root = root.parent
>>>>>>> parent of 542ad17 (time module)
=======

        while root.parent is not None:
            root = root.parent
>>>>>>> parent of 542ad17 (time module)

        bestNode = self.findBestUCT(root)
        #print("the node i get", bestNode.get_pos(), bestNode.get_dir())

        my_pos = bestNode.get_pos()
        dir = bestNode.get_dir()

        #print("---------------------------------------------------the node that I returned--------------------------------------------------", my_pos, dir)
        return my_pos, dir

    def check_valid_step(self, chess_board, start_pos, end_pos, adv_pos, barrier_dir, step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).
        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True
        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:

            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached

    # get all actions under the root Node
    def getActions(self, rootNode, my_pos, adv_pos, step, chess_board):
        # boardsize chess_board.boardsize
        x = chess_board[0].shape[0]
        actions = []
        uPos = []
        for r in range(step + 1):
            c = step - r
            for i in range(c + 1):  # 3 is the size of the chessboard
                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] + i) <= x - 1:
                    next_pos = (my_pos[0] + r, my_pos[1] + i)
                    if next_pos not in uPos:
                        uPos.append(next_pos)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] + i) <= x - 1:
                    next_pos2 = (my_pos[0] - r, my_pos[1] + i)
                    if next_pos2 not in uPos:
                        uPos.append(next_pos2)

                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] - i) <= x - 1:
                    next_pos3 = (my_pos[0] + r, my_pos[1] - i)
                    if next_pos3 not in uPos:
                        uPos.append(next_pos3)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] - i) <= x - 1:
                    next_pos4 = (my_pos[0] - r, my_pos[1] - i)
                    if next_pos4 not in uPos:
                        uPos.append(next_pos4)
            if (c == 0):
                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] + c) <= x - 1:
                    next_pos = (my_pos[0] + r, my_pos[1] + i)
                    if next_pos not in uPos:
                        uPos.append(next_pos)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] + c) <= x - 1:
                    next_pos2 = (my_pos[0] - r, my_pos[1] + i)
                    if next_pos2 not in uPos:
                        uPos.append(next_pos2)

                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] - c) <= x - 1:
                    next_pos3 = (my_pos[0] + r, my_pos[1] - i)
                    if next_pos3 not in uPos:
                        uPos.append(next_pos3)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] - c) <= x - 1:
                    next_pos4 = (my_pos[0] - r, my_pos[1] - i)
                    if next_pos4 not in uPos:
                        uPos.append(next_pos4)


        if len(uPos) != 0:
            for z in uPos:
                for i in range(4):
                    if (self.check_valid_step(chess_board, my_pos, z, adv_pos, i, step)):
                        n1 = Node(z, i, rootNode)
                        n1.set_level(rootNode.get_level() + 1)
                        actions.append(n1)


        else:
            actions = []

        return actions

    # set UCT score to the list of children actions Nodes
    def calculateUCT(self, node):
        children = node.get_children()
        actions = list(children.keys())

        for i in range(len(actions)):
            if (actions[i].get_visits() == 0):
                valueA = math.inf
                uct = math.inf
                children[actions[i]] = uct
                break
            else:
                valueA = actions[i].get_wins() / actions[i].get_visits()
                numA = actions[i].get_visits()
            visitR = node.get_visits()
            if (visitR == 0):
                uct = math.inf
                children[actions[i]] = uct
            else:
                uct = valueA + math.sqrt(2) * math.sqrt(math.log(visitR, np.e) / numA)
                children[actions[i]] = uct
        return children

    # find the best node to expand
    def findBestUCT(self, rootNode):
        # find all children of this current node
        rootChildren = rootNode.get_children()
        bestNode = None
        bestUCT = 0
        # calculate UCTs for all children of this node
        self.calculateUCT(rootNode)
        # iterate through all children of this node
        #for key, value in rootChildren.items():
            #print("children, uct value", key.get_pos(), key.get_dir(), value)
        for key, value in rootChildren.items():

            if value == math.inf:
                # print("best uct is infinite")
                return key
            elif value > bestUCT:

                bestUCT = value
                bestNode = key
        #print("bestNode is", bestNode.get_pos(), bestNode.get_dir())
        #print("best uct is", bestUCT)
        return bestNode

    def select(self, rootNode,
               chess_board):  # start from the root node, select its children until leaf(uct can't decide)\
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # if the node has children, put barrier
        while len(rootNode.get_children()) != 0:
            # for key, value in rootNode.get_children().items():
            # print("every children printed in the select pro", key.get_pos(), key.get_dir())
            # descend a layer deeper
            dir = rootNode.get_dir()
            r, c = rootNode.get_pos()
            if (dir is not None):
                chess_board[r, c, dir] = True
                move = moves[dir]
                chess_board[r + move[0], c + move[1], opposites[dir]] = True
            # move to the next layer
            rootNode = self.findBestUCT(rootNode)

        dir2 = rootNode.get_dir()
        r2, c2 = rootNode.get_pos()

        if (dir2 is not None):
            chess_board[r2, c2, dir2] = True
            move = moves[dir2]
            chess_board[r2 + move[0], c2 + move[1], opposites[dir2]] = True

        return rootNode

    def expand(self, leafNode, my_pos, adv_pos, step, chess_board):  # add all children to the leaf node
        AllChildren = self.getActions(leafNode, my_pos, adv_pos, step, chess_board)

        children = dict()
        for i in range(len(AllChildren)):
            children[AllChildren[i]] = math.inf
        leafNode.set_children(children)

    def updateNode(self, n, totalVisits, numWins):
        newVisit = n.get_visits() + totalVisits
        n.set_visits(newVisit)
        newWins = n.get_wins() + numWins
        n.set_wins(newWins)

    def backpropagation(self, lastExpand, totalVisits, numWins):
        while lastExpand.get_parent() is not None:
            self.updateNode(lastExpand, totalVisits, numWins)
            # print("1 node visited is:", lastExpand.get_visits(), "1 node wins is", lastExpand.get_wins())
            lastExpand = lastExpand.get_parent()
        self.updateNode(lastExpand, totalVisits, numWins)
        return lastExpand

    def m_simulate(self, chess_board, my_pos, adv_pos, tree_self_turn):
        #print("in m simulation, my_pos is", my_pos)
        #print("in m simulation, adv_pos is", adv_pos)
        #print("in tree_self_turn, self turn is", tree_self_turn)
        i = 0
        # declare the global variables
        totalS = 0
        while (i != 5):  # check time,
            tempC = deepcopy(chess_board)
            default_self_turn = deepcopy(tree_self_turn)
            # random initialize p0 and p1 position, remember to change it after expansion
            # print("chessboard before each simulation should be equal to selected chessboard", chess_board)
            success = self.run_simulation(tempC, my_pos, adv_pos, default_self_turn)
            if success == 1:
                totalS = totalS + 1
            i += 1

        return i, totalS

    def random_walk(self, my_pos, adv_pos, max_step, chess_board):
        # print("take random walk")
        """
        Randomly walk to the next position in the board.
        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
        # Random Walk

        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)
            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

            # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        # print("random walk",my_pos,chess_board[r, c])

        x = 0
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
            x = x + 1

        return my_pos, dir

    def check_endgame(self, board_size, chess_board, p0_pos, p1_pos):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score

    def simulation_step(self, chess_board, my_pos, adv_pos, self_turn):
        # print("chess_board before updated", chess_board)

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        board_size = chess_board[0].shape[0]
        max_step = (board_size + 1) // 2

        # adv is p0
        p0_pos = deepcopy(adv_pos)
        # my is p1
        p1_pos = deepcopy(my_pos)

        x = deepcopy(chess_board)

        # if adv play
        if not self_turn:
            # cur_pos = adv and adv_pos = my
            cur_pos = np.asarray(p0_pos)
            adv_pos = p1_pos

        else:
            cur_pos = np.asarray(p1_pos)
            adv_pos = p0_pos
        next_pos, dir = self.random_walk(tuple(cur_pos), tuple(adv_pos), max_step, x)
        next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        # adv turn, next_pos is for adv

        if not self_turn:
            p0_pos = next_pos
        # my turn, next_pos is for me
        else:
            p1_pos = next_pos

        # Set the barrier to True

        r, c = next_pos
        chess_board[r, c, dir] = True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True
        # print("each simulation step chessboard", chess_board)

        # Change turn
        self_turn = 1 - self_turn

        # print("before check_endgame should be all true", chess_board[r,c])
        # print("my position is", next_pos)
        results = self.check_endgame(board_size, chess_board, p0_pos, p1_pos)

        return next_pos, results, self_turn
        # remember to return the updated chessboard

    def run_simulation(self, chess_board, my_pos, adv_pos, self_turn):
        r, c = my_pos
        if chess_board[r, c, 0] and chess_board[r, c, 1] and chess_board[r, c, 2] and chess_board[r, c, 3]:
            return -1

        result = self.simulation_step(chess_board, my_pos, adv_pos, self_turn)
        is_end, p0_score, p1_score = result[1]
        self_turn = result[2]

        if not self_turn:
            my_pos = result[0]
        else:
            adv_pos = result[0]

        while not is_end:
            # print("game not terminate")
            result = self.simulation_step(chess_board, my_pos, adv_pos, self_turn)
            is_end, p0_score, p1_score = result[1]
            self_turn = result[2]
            if not self_turn:
                my_pos = result[0]
            else:
                adv_pos = result[0]

        # print("game terminates")

        if p0_score > p1_score:  # adversary wins
            return -1
        elif p0_score < p1_score:  # I win
            return 1
        else:  # it's a tie
            return 0.5


'''
def main():
    # initialize chessboard
    x = np.zeros((4, 4, 4), dtype=bool)
    x[0, :, 0] = True
    x[:, 0, 3] = True
    x[-1, :, 2] = True
    x[:, -1, 1] = True
    chess_board = x


    # given my position and adversary position
    my_pos = (2, 2)
    adv_pos = (0, 1)

    # max step is given
    step = 2

    sa = StudentAgent()

    # initialize root node
    root = Node(my_pos, None, None)


    # test for find all children

    children = sa.getActions(root, adv_pos, 2, x)
    for i in range(len(children)):
        print(children[i].my_pos, children[i].dir)


    #test for select
    sa.select(root, chess_board)
    print(root.get_pos())


    # test for expand
    sa.expand(root, adv_pos, step, chess_board)
    children = root.get_children()
    for key, value in children.items():
        print("in loop")
        print(key.get_pos(), key.get_dir(), value)


    # test for simulation
    self_turn = 0
    # random initialize p0 and p1 position, remember to change it after expansion
    p0_pos = [0, 0]
    p1_pos = [1, 1]
    print(sa.run_simulation(x, my_pos, adv_pos, p0_pos, p1_pos, self_turn))



    # test for backpropagation

    success = sa.m_simulate(chess_board, root, adv_pos)
    sa.backpropagation(root, success)
    print("it is visited:", root.get_visits(), "num of success", root.get_wins())
    #test for the game logic with find best UCT

    #1. the root is not visited and not simulated before
    totalvisits, numwins = sa.m_simulate(chess_board, root, adv_pos)
    returned_node = sa.backpropagation(root, totalvisits, numwins)
    print("root node is:", root.get_pos(), "returned node is:", returned_node.get_pos())
    print("returned node is visited:", returned_node.get_visits(),"returned node wins:", returned_node.get_wins())


    #2. the root is simulated and visited, but is the leaf node, so expand the game tree
    sa.select(root,chess_board)
    sa.expand(root, adv_pos, step, chess_board)
    children = root.get_children()
    for key, value in children.items():
        print(key.get_pos())


    #3. the root is no longer a leaf node, find the best uct among its children
    cur_node = sa.findBestUCT(root)
    print("root node is:", root.get_pos(), root.get_dir(),"current node is:", cur_node.get_pos(), cur_node.get_dir())

    #4. move to the next layer, if the current node has not been visited, simulate
    if not cur_node.get_children():
        print("cur node no children")
    if cur_node.get_visits() == 0:
        totalVisits, numWins = sa.m_simulate(chess_board,cur_node, adv_pos)

        #5. backpropagation of the success rate and to see whether the root node is returned
        returned_node = sa.backpropagation(cur_node, totalVisits, numWins)
        print("the returned node is:", returned_node.get_pos(), returned_node.get_dir(),
              "its total visits is:", returned_node.get_visits(), "its total wins is:", returned_node.get_wins())

    #6. the current node is now visited, but since it is a leaf node, it need to be expanded
    sa.select(returned_node, chess_board)
    sa.expand(returned_node, adv_pos, step, chess_board)
    childrenR = returned_node.get_children()
    for key, value in childrenR.items():
        print(key.get_pos(), key.get_dir())
    #7. the current node is now visited, and no longer a leaf node. move to the next layer
    third_node = sa.findBestUCT(returned_node)

    #8 move to the next 




    #test for findBestUCT
    while cur_node.get_parent() is not None:
        cur_node = cur_node.get_parent()
    best_node = sa.findBestUCT(cur_node)
    print("the root's best UCT is ")


if __name__ == "__main__":
    main()
'''