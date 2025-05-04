import tkinter as tk
import time
import numpy as np

from tessellate_game import GameState, RED, BLUE

PREVIEW, NO_PREVIEW = 0, 1

#formatting constants
bg_color = 'grey'
bg_tag = 'background'
board_outline_color = '#0de50d'
color_map = {RED : {False : 'red', True : '#F87272'}, BLUE : {False : 'blue', True : '#94A8F8'}}
num_squares = 5
line_width = 15
window_size = 600
sq_size = int(window_size/num_squares)


class GameWindow(tk.Tk):
	# Class to handle the tkinter logic of drawing shapes and handling the mouse behavior
	
	def __init__(self):
		#initializes the GameWindow class (inheriting from the tkinter base class)
		tk.Tk.__init__(self)

		self.setup_canvas()

		# boolean variable, controls whether the game should draw a temporary tile, to simulate the mouse "hovering" over the board
		# depending on the current coordinates of the mouse
		# the design is intended to only display the preview in locations on the board where location of the user's intended click is unambiguous
		self.display_state = NO_PREVIEW

		# stores a reference reference to the tile being current displayed while the mouse is hovering
		self.current_display = None

		# stores the board information as defined by the GameState class above
		self.game_state = GameState()

		self.draw_base()
		
	def setup_canvas(self):
		#Formats the tkinter canvas

		#add label to displays the scores at the top
		self.score_label = tk.Label(text = 'red : 1     blue : 1', font = ("Helvetica", 20))
		self.score_label.pack()

		#create Canvas where we draw the game board
		self.canvas = tk.Canvas(height = window_size, width = window_size)
		self.canvas.pack()

		#equip canvas with a mouse listener, calls the function respond_to_click below
		self.canvas.tag_bind(bg_tag, '<Button-1>', self.respond_to_click)
		self.bind('<Motion>', self.motion)
	
	def motion(self,event):
		# Updates the game and display information whenever the user moves their mouse

		# When the user is hovering over a location they are allowed to play on, a preview of the move is displayed to the board
		# Then, that preview tile is deleted from the canvas when the user moves their mouse away to a new spot

		#gets the coordinate the user clicked at
		result = self.clickable_coord(event.x, event.y)

		if result:
			row, col = result

			if self.display_state == NO_PREVIEW:

				self.display_state = PREVIEW
				self.current_display = result
				self.temp_tile = self.draw_tile(row, col, preview = True)

			elif self.display_state == PREVIEW and self.current_display != result:

				self.canvas.delete(self.temp_tile)
				self.current_display = result
				self.temp_tile = self.draw_tile(row, col, preview = True)

		elif self.display_state == PREVIEW:
			self.canvas.delete(self.temp_tile)
			self.display_state = NO_PREVIEW


	def respond_to_click(self, event):
		# Updates the window and game information whenever the user clicks their mouse

		result = self.clickable_coord(event.x, event.y)

		if result:

			#get the coordinates where the user clicked
			row, col = result

			#draw tile on the board
			self.draw_tile(row, col)

			# update the information in the GameState object
			self.game_state.new_move(row, col)

			#get the current score
			red_score, blue_score = self.game_state.get_score()

			#format current score onto the window
			self.score_label['text'] = "red : {}     blue : {}".format(red_score, blue_score)	

	def clickable_coord(self, x, y):
		# Returns the grid coordinate where the user clicked, but only if the row is clickable

		# a location (x, y) is clickable if it passes the following 3 checks

		# 1) the user has clicked on the game board
		# 2) the user has clicked near a corner of a grid square
		# 3) there is currently no existing game piece where the user has clicked

		# check 1) the user has clicked on the game board
		off_the_board = (x < 0 or x >= window_size or y < 0 or y >= window_size)
		if not off_the_board:

			# check 2) the user has clicked near a corner of a grid square
			xx, yy = x % sq_size, y % sq_size
			in_a_corner = not ((xx > sq_size//3 and xx < 2*sq_size//3) or (yy > sq_size//3 and yy < 2*sq_size//3))
			if in_a_corner:

				# check 3) there is currently no existing game piece where the user has clicked
				row, col = y // (window_size // 10), x // (window_size // 10)

				if self.game_state.playable(row, col):

					return (row, col)
	
	def draw_base(self):
		# draws underlying board

		self.canvas.create_rectangle(0, 0, window_size, window_size, fill = bg_color, tag = bg_tag) #background
		for i in range(1, num_squares): #white lines
			self.canvas.create_line(i*sq_size, line_width, i*sq_size, window_size - line_width, width = 5, fill = 'white')
			self.canvas.create_line(line_width, i*sq_size, window_size - line_width, i*sq_size, width = 5, fill = 'white')

	def draw_tile(self, row, col, preview = False):
		# draws a triangular tile onto the board based on the row, col of the users move
		# preview: boolean variable that tells whether the tile is a move, or just the preview of a move
		# if preview == True, use lighter color, else use darker color to draw tile

		# formula to convert the row, col coordinates into the coordinates of its cooresponding triangle on the board
		x, xx, y, yy = col//2, col % 2, row//2, row % 2
		points = [p*sq_size for p in [x + xx, y + yy, x + 1 - xx, y + yy, x + xx, y + 1 - yy]]


		return self.canvas.create_polygon(points, outline = bg_color, fill = color_map[self.game_state.turn][preview], tag = bg_tag)
