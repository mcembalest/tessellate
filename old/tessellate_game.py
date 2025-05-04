from sys import maxsize as inf
import time
import numpy as np

#constants to represent each game piece
RED, BLUE, EMPTY, OUT_OF_PLAY = 0, 1, 2, 3


class GameState:
	# Class to handle the game logic

	#def __init__(self, against_computer = False):
	def __init__(self):
		# the game board is a 10x10 matrix
		# possible values for entries in the game board are EMPTY, RED, BLUE, OUT_OF_PLAY
		self.board = np.array([[EMPTY for i in range(10)] for j in range(10)])

		# switches between RED := 0, BLUE := 1
		self.turn = RED

		self.score = {RED : 1, BLUE : 1}
		self.num_islands = {RED : 0, BLUE : 0}

		self.legal_moves = {(i, j) : True for i in range(10) for j in range(10)}

	def new_move(self, row, col):

		if not self.playable(row, col):
			print('uh oh', row, col)
			print(self.board)
			print('legal', [x for x in self.legal_moves if self.legal_moves[x]==1])
			raise ValueError('ILLEGAL MOVE')

		# add the tile to the board
		self.add_tile(row, col)

		# switch the current player
		self.turn = 1 - self.turn

		self.calculate_score()

	def add_tile(self, row, col):
		# add a tile to the board, and prevent invalid tiles from being clicked on in the future

		self.board[row][col] = self.turn
		self.board[row][col + (-1)**col] = OUT_OF_PLAY
		self.board[row + (-1)**row][col] = OUT_OF_PLAY

		self.legal_moves[(row, col)] = 0
		self.legal_moves[(row, col + (-1)**col)] = 0
		self.legal_moves[(row + (-1)**row, col)] = 0

	# def unmake_move(self, row, col):
	# 	# make the space at (row, col) set to the value EMPTY
	# 	# also, reverse any tiles that would have been labelled OUT_OF_PLAY by the tile at (row, col)
	# 	self.board[row][col] = EMPTY
	# 	self.legal_moves[(row, col)] = 1
	# 	if self.board[row + (-1)**row][col + (-1 )**col] == EMPTY:
	# 		self.board[row][col + (-1)**col] = EMPTY
	# 		self.board[row + (-1)**row][col] = EMPTY

	# 		self.legal_moves[(row, col + (-1)**col)] = 1
	# 		self.legal_moves[(row + (-1)**row, col)] = 1
	# 	self.turn = 1 - self.turn

	def calculate_score(self):
		# Returns the score for each player
		# A player's score is defined as the product of all their current island sizes
		# This function uses dfs(...) to determine the size of each island

		self.score = {RED : 1, BLUE : 1}
		self.num_islands = {RED : 0, BLUE : 0}
		visited = [[False for j in range(10)] for i in range(10)]
		for row0 in range(10):
			for col0 in range(10):
				color = self.board[row0][col0]
				if color in [RED, BLUE] and not visited[row0][col0]:
					size = self.dfs(row0, col0, color, visited)
					self.score[color] *= size
					if size > 1:
						self.num_islands[color] += 1

	def get_score(self):
		return self.score[RED], self.score[BLUE]

	def dfs(self, row0, col0, color, visited):
		# helper function for calculate_score
		# Returns the number of reachable nodes from starting point (row0, col0) using depth-first search

		size = 0
		stack = [(row0, col0)]
		while stack:
			row, col = stack.pop(0)
			if not visited[row][col] and self.board[row][col] == color:
				size += 1
				visited[row][col] = True
				stack += self.neighbors(row, col)
		return size

	def neighbors(self, row, col):
		#Returns a list of the coordinates of the neighbors of (row, col)

		#the (row, col) coordinates of any neighboring piece
		#sorry for the annoyingly long formula...think of it as a challenge to see for yourself why it works :D
		neighbor_coords = [(row + (-1)**row, col + (-1 )**col), (row - 1, col - (-1)**(row + col + 1)), (row + 1, col + (-1)**(row + col + 1)), (row + (-1)**(row + 1), col), (row, col + (-1)**(col + 1))]
		
		#use filter to remove any neighboring coordinates that fall off the board
		return list(filter(lambda p : p[0] in range(10) and p[1] in range(10), neighbor_coords))

	def playable(self, row, col):
		# Returns True if and only if there is no tile currently at (row, col)
		return self.board[row][col] == EMPTY


	