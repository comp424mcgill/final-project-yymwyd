# Student agent: Add your own agent here
# from agents.agent import Agent
import numpy as np
from Node import *
from agent import Agent
import math
from copy import deepcopy
import copy
import sys
from store import register_agent


@register_agent("student_agent")
class agentTest(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(agentTest, self).__init__()
        self.name = "testAgent"
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
        root = Node(my_pos, None, None)
        endNode = None
        bestNode = None
        while True:
            if not root.get_children():
                if root.get_visits == 0:
                    self.m_simulate(chess_board, root, adv_pos)
                else:
                    leafNode = self.select(root)
                    root = leafNode
                    endNode = self.expand(root, adv_pos, max_step, chess_board)

                    if endNode:
                        break

            else:
                root = self.findBestUCT(root)
        while leafNode.parent:
            leafNode = leafNode.parent
        root = leafNode
        bestNode = self.findbestUCT(root)

        my_pos = bestNode.get_pos()
        dir = bestNode.get_dir()

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
    def getActions(self, rootNode, adv_pos, step, chess_board):
        # boardsize chess_board.boardsize
        x = 4
        y = 4
        my_pos = rootNode.get_pos()
        r = 0
        actions = []
        uPos = []
        for r in range(step + 1):
            c = step - r
            for i in range(c + 1):  # 3 is the size of the chessboard

                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] + i) <= x - 1:
                    next_pos = (my_pos[0] + r, my_pos[1] + i)

                if next_pos not in uPos and not (next_pos == my_pos):
                    uPos.append(next_pos)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] + i) <= x - 1:
                    next_pos2 = (my_pos[0] - r, my_pos[1] + i)

                if next_pos2 not in uPos and not (next_pos2 == my_pos):
                    uPos.append(next_pos2)

                if 0 <= (my_pos[0] + r) <= x - 1 and 0 <= (my_pos[1] - i) <= x - 1:
                    next_pos3 = (my_pos[0] + r, my_pos[1] - i)

                if next_pos3 not in uPos and not (next_pos3 == my_pos):
                    uPos.append(next_pos3)

                if 0 <= (my_pos[0] - r) <= x - 1 and 0 <= (my_pos[1] - i) <= x - 1:
                    next_pos4 = (my_pos[0] - r, my_pos[1] - i)
                if next_pos4 not in uPos and not (next_pos4 == my_pos):
                    uPos.append(next_pos4)

        for z in uPos:
            for i in range(4):
                if (self.check_valid_step(chess_board, my_pos, z, adv_pos, i, step)):
                    n1 = Node(z, i, my_pos)
                    actions.append(n1)
        return actions

    # set UCT score to the list of children actions Nodes
    def setUCT(self, node):
        children = node.get_children()
        actions = children.keys()
        for i in range(len(actions)):
            valueA = actions[i].get_wins() / actions[i].get_visit()
            numA = actions[i].get_visit * ()
            visitR = node.get_visit()
            uct = valueA + math.sqrt(2) * math.sqrt(math.log(visitR, np.e) / numA)
            children[actions[i]] = uct
        return children

    # find the best node to expand
    def findBestUCT(self, rootNode):
        # find all children of this current node
        rootChildren = rootNode.get_children()
        bestNode = None
        bestUCT = 0
        # for all children of this node, return the ones that either has the infinite UCT
        for key, value in rootChildren.items():
            if value == math.inf:
                return key
            # or the largest one
            elif value > bestUCT:
                bestNode = key
        return bestNode

    def select(self, rootNode,
               chess_board):  # start from the root node, select its children until leaf(uct can't decide)\
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        while rootNode.get_children():
            # descend a layer deeper
            rootNode = self.findBestUCT(rootNode)
            dir = rootNode.get_dir()
            r, c = rootNode.get_pos()
            chess_board[r, c, dir] = True
            move = moves[dir]
            chess_board[r + move[0], c + move[1], opposites[dir]] = True
        return rootNode

    def expand(self, leafNode, adv_pos, step, chess_board):  # add all children to the leaf node
        AllChildren = self.getActions(leafNode, adv_pos, step, chess_board)
        if not AllChildren:
            return leafNode
        children = dict()
        for i in range(len(AllChildren)):
            children[AllChildren[i]] = math.inf
        leafNode.set_children(children)

    def updateNode(self, n, success):
        newVisit = n.get_visits() + 1
        n.set_visits(newVisit)
        if (success):
            newWin = n.get_wins() + 1
            n.set_wins(newWin)

    def backpropagation(self, lastExpand, success):
        while lastExpand.parent != None:
            self.updateNode(lastExpand, success)
            lastExpand = lastExpand.parent
        self.setUCT(lastExpand)

    def rollout(self, rootNode, adv_pos, step, chess_board):
        leafNode = self.select(rootNode)
        self.expand(leafNode, adv_pos, step, chess_board)
        if leafNode.get_children():
            self.m_simulate(chess_board, leafNode, adv_pos, step)

    def m_simulate(self, chess_board, leafNode, adv_pos):
        my_pos = leafNode.get_pos()
        i = 0
        # declare the global variables

        while (i != 20):  # check time
            tempC = deepcopy(chess_board)
            self_turn = 0
            # random initialize p0 and p1 position, remember to change it after expansion
            p0_pos = [0, 0]
            p1_pos = [1, 1]
            success = self.run_simulation(tempC, my_pos, adv_pos, p0_pos, p1_pos, self_turn)
            self.backpropagation(leafNode, success)
            i += 1
        return leafNode

    def random_walk(self, my_pos, adv_pos, max_step, chess_board):
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
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    def check_endgame (self, board_size, chess_board, p0_pos, p1_pos):
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

    def simulation_step(self, chess_board, my_pos, adv_pos, p0_pos, p1_pos, self_turn):

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        board_size = chess_board[0].shape[0]
        max_step = (board_size + 1) // 2
        # global self_turn
        # global p0_pos
        # global p1_pos

        # adv is p0
        p0_pos = deepcopy(adv_pos)
        # my is p1
        p1_pos = deepcopy(my_pos)

        # if adv play
        if not self_turn:
            # cur_pos = adv and adv_pos = my
            cur_pos = np.asarray(p0_pos)
            adv_pos = p1_pos

        else:
            cur_pos = np.asarray(p1_pos)
            adv_pos = p0_pos

        next_pos, dir = self.random_walk(tuple(cur_pos), tuple(adv_pos), max_step, chess_board)
        next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        # adv turn, next_pos is for adv
        print("p0: ", p0_pos)
        print("p1 ", p1_pos)
        if not self_turn:
            p0_pos = next_pos
        # my turn, next_pos is for me
        else:
            p1_pos = next_pos

        # Set the barrier to True
        r, c = next_pos
        print("p0: ", p0_pos)
        print("p1 ", p1_pos)
        print(next_pos, chess_board[r, c])
        chess_board[r, c, dir] = True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True
        print(next_pos, chess_board[r, c])

        # Change turn
        self_turn = 1 - self_turn

        results = self.check_endgame(board_size, chess_board, p0_pos, p1_pos)
        print("the game is end ", results[0])

        return next_pos, results, self_turn
        # remember to return the updated chessboard

    def run_simulation(self, chess_board, my_pos, adv_pos, p0_pos, p1_pos, self_turn):
        # global p0_pos
        # global p1_pos

        result = self.simulation_step(chess_board, my_pos, adv_pos, p0_pos, p1_pos, self_turn)
        is_end, p0_score, p1_score = result[1]
        self_turn = result[2]

        if not self_turn:
            my_pos = result[0]
        else:
            adv_pos = result[0]

        while not is_end:
            result = self.simulation_step(chess_board, my_pos, adv_pos, p0_pos, p1_pos, self_turn)
            is_end, p0_score, p1_score = result[1]
            self_turn = result[2]
            if not self_turn:
                my_pos = result[0]
            else:
                adv_pos = result[0]

        if p0_score > p1_score:  # adversary wins
            return -1
        elif p0_score < p1_score:  # I win
            return 1
        else:  # it's a tie
            return 0.5


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

    sa = agentTest()

    # initialize root node
    root = Node(my_pos, None, None)

    # test for find all children
    '''
    children = sa.getActions(root, adv_pos, 2, x)
    for i in range(len(children)):
        print(children[i].my_pos, children[i].dir)
    '''

    # test for select
    '''
    sa.select(root)
    print(root.get_pos())
    '''

    # test for expand
    '''
    sa.expand(root, adv_pos, step, chess_board)
    children = root.get_children()
    for key, value in children.items():
        print("in loop")
        print(key.get_pos(), key.get_dir(), value)
    '''

    # test for backpropagation

    # test for simulation
    self_turn = 0
    # random initialize p0 and p1 position, remember to change it after expansion
    p0_pos = [0, 0]
    p1_pos = [1, 1]
    print(sa.run_simulation(x, my_pos, adv_pos, p0_pos, p1_pos, self_turn))


if __name__ == "__main__":
    main()
