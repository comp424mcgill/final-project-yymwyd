import numpy as np
from copy import deepcopy

def random_walk(my_pos, adv_pos,max_step, chess_board):
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
            print("intial wish of position: ",my_pos)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                if(my_pos == adv_pos):
                        print("next position has been taken by adversary!")
                else:
                        print("enclosed by all direction!")
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)
                print("changed position after adversary is token: ",my_pos)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        #print("random walk",my_pos,chess_board[r, c])
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

def check_endgame(board_size,chess_board,p0_pos,p1_pos):
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


#declare the global variables
self_turn = 0

p0_step=0
p1_step=0

#random initialize p0 and p1 position, remember to change it after expansion
p0_pos = [0,0]
p1_pos = [1,1]

def simulation_step(chess_board, my_pos, adv_pos):
        
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        board_size = chess_board[0].shape[0]
        max_step = (board_size +1) //2
        global self_turn
        global p0_pos
        global p1_pos


        #adv is p0
        p0_pos = deepcopy(adv_pos)
        #my is p1
        p1_pos = deepcopy(my_pos)


        #if adv play
        if not self_turn:
            #cur_pos = adv and adv_pos = my
                cur_pos=  np.asarray(p0_pos)
                adv_pos = p1_pos

        else:
                cur_pos=  np.asarray(p1_pos)
                adv_pos = p0_pos


        next_pos, dir = random_walk(tuple(cur_pos), tuple(adv_pos),max_step,chess_board)
        next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        print("p0 pos ", p0_pos)
        print("p1pos ", p1_pos)
        #adv turn, next_pos is for adv
        if not self_turn:
                    p0_pos = next_pos
        #my turn, next_pos is for me
        else:
                    p1_pos = next_pos
        print("p0 pos ", p0_pos)
        print("p1pos ", p1_pos)

        # Set the barrier to True
        r, c = next_pos

        print(next_pos,chess_board[r, c])
        chess_board[r, c, dir] = True
        move = moves[dir]
        chess_board[r+ move[0], c + move[1], opposites[dir]] = True
        print(next_pos,chess_board[r, c])


        # Change turn
        self_turn = 1 - self_turn

        results = check_endgame(board_size,chess_board,p0_pos,p1_pos)

        return next_pos,results#,self_turn
        #remember to return the updated chessboard

def run_simulation(chess_board, my_pos, adv_pos):
        global p0_pos
        global p1_pos
        
        result = simulation_step(chess_board, my_pos, adv_pos)
        is_end, p0_score,p1_score = result[1]
        #self_turn = result[2]
        
        if not self_turn:
                    my_pos = result[0]
        else:
                    adv_pos = result[0]
                    
        while not is_end:
                result = simulation_step(chess_board, my_pos, adv_pos)
                is_end, p0_score,p1_score = result[1]
                if not self_turn:
                        my_pos = result[0]
                else:
                        adv_pos = result[0]
              
        if p0_score >p1_score:
                print("p0 wins"+" p0 score: "+ str(p0_score)+" p1 score: "+str(p1_score))
        elif p0_score < p1_score:
                print( "p1 wins"+" p0 score: "+ str(p0_score)+" p1 score: "+str(p1_score))
        else:
                print( "it's a tie"+" p0 score: "+ str(p0_score)+" p1 score: "+str(p1_score))
                
        return p0_score, p1_score

def tn_run(chess_board, my_pos, adv_pos, max_step):
        i=0
        while(i != 10):
                p0_score, p1_score = run_simulation(chess_board,my_pos, adv_pos)
                i = i+1;
                print("---------finish",i, "th", "simulation------")

#multiple simulations
def m_simulate(chess_board, my_pos, adv_pos, max_step):
        p1_win_count = 0
        p2_win_count = 0
        i =0
        while (i != 10):
                p0_score, p1_score = run_simulation(chess_board,my_pos, adv_pos)
                i +=1
                
                if p0_score > p1_score:
                        p1_win_count += 1
                elif p0_score < p1_score:
                        p2_win_count += 1
                else:  # Tie
                        p1_win_count += 1
                        p2_win_count += 1
                       
                        
        return p1_win_count,i,"---------------------------------------------------------"
        
        
            
if __name__ == "__main__":
        chess_board = np.zeros((4,4, 4), dtype=bool)
        chess_board[0, :, 0] = True
        chess_board[:, 0, 3] = True
        chess_board[-1, :, 2] = True
        chess_board[:, -1, 1] = True
        max_step = 2

        my_pos = (2,3)
        adv_pos = (0,1)
        #print(chess_board)
        print(run_simulation(chess_board, my_pos, adv_pos))
        


        
