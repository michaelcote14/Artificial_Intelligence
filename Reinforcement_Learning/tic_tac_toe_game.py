import random
import time

class tic_tac_toe:
    def __init__(self):
        # Step 1: print the game board
        self.board = ["-", "-", "-",
                    "-", "-", "-",
                    "-", "-", "-"]
        self.currentPlayer = 'X' # X goes first
        self.winner = None
        self.is_game_over = False

    def reset(self):
        # Step 1: print the game board
        self.board = ["-", "-", "-",
                    "-", "-", "-",
                    "-", "-", "-"]
        self.currentPlayer = 'X' # X goes first
        self.winner = None
        self.is_game_over = False

    def perform_move(self, neural_move):
        self.currentPlayer = 'X'

        # Make the move using the neural network
        self.board[neural_move] = self.currentPlayer

        self.checkForWin(self.board)
        self.checkForTie(self.board)
        self.switch_player()
        self.computer(self.board)
        self.checkForWin(self.board)
        self.checkForTie(self.board)

        # Check for win or tie
        if self.winner == 'X':
            neural_reward = 1
        elif self.winner == 'O':
            neural_reward = -1
        else:
            neural_reward = 0.0

        return neural_reward, self.is_game_over, self.winner

    def displayBoard(self, board):
        print("\n")
        print(board[0] + " | " + board[1] + " | " + board[2])
        print("---------")
        print(board[3] + " | " + board[4] + " | " + board[5])
        print("---------")
        print(board[6] + " | " + board[7] + " | " + board[8])

    # Step 3: Check for win or tie
    def checkForTie(self, board):
        if "-" not in board:
            self.is_game_over = True

    def checkForWin(self, board):
        if self.checkHorizontal(board) or self.checkVertical(board) or self.checkDiagonal(board):
            self.is_game_over = True
    def checkHorizontal(self, board):
        if board[0] == board[1] == board[2] != "-":
            self.winner = board[0]
            return True
        elif board[3] == board[4] == board[5] != "-":
            self.winner = board[3]
            return True
        elif board[6] == board[7] == board[8] != "-":
            self.winner = board[6]
            return True
        else:
            return False

    def checkVertical(self, board):
        if board[0] == board[3] == board[6] != "-":
            self.winner = board[0]
            return True
        elif board[1] == board[4] == board[7] != "-":
            self.winner = board[1]
            return True
        elif board[2] == board[5] == board[8] != "-":
            self.winner = board[2]
            return True
        else:
            return False

    def checkDiagonal(self, board):
        if board[0] == board[4] == board[8] != "-":
            self.winner = board[0]
            return True
        elif board[2] == board[4] == board[6] != "-":
            self.winner = board[2]
            return True

    # Step 4: Switch whose turn it is
    def switch_player(self):
        if self.currentPlayer == 'X':
            self.currentPlayer = 'O'
        else:
            self.currentPlayer = 'X'

    # Step 6: make a computer player
    def computer(self, board):
        while self.currentPlayer == "O" and self.is_game_over == False:
            position = random.randint(0, 8)
            if board[position] == "-":
                board[position] = "O"
                self.switch_player()

            if self.checkForWin(board):
                self.is_game_over = True
            elif self.checkForTie(board):
                self.is_game_over = True


if __name__ == '__main__':
    my_game = tic_tac_toe()
    my_game.reset()