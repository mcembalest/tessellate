This game was created by accident. This was possible because I carried around a small
notebook and a pen in my pocket.

Tessellate

Goals:

determine the optimal play for this game
build an automated agent who can play at various skill levels
Figure 1a Figure 1b

Two players, red and blue, place triangular tiles on the board one after another until the board is
full. The triangles can be placed along either diagonal when a square is unoccupied, but must
not overlap with any existing tiles already placed on the board.

The score at the end of the game is the product ofthe sizes of regions you created during the
game. For example, in Figure 1a, the red player has created four regions of triangles, whose
sizes are 17, 5, 2, and 1, leading to a score of 17521 = 170. Likewise, in Figure 1b, the blue
player has created nine regions of tiles, whose sizes are 5, 5, 3, 3, 3, 2, 2, 1, and 1, leading to a
score of 55333221*1 = 2700.

The score of the game can shift drastically if, for example, a player with a previously high score
suddenly lowers it by placing a tile that merges regions together. For example, the red player
lowers their score by joining two regions from Figure 2a to 2b, and the blue player lowers their
score by joining three regions from Figure 2c to 2d.

Figure 2a → Figure 2b

Figure 2c → Figure 2d

Strategy # 1

Model:

State
The current board configuration
Action
The integer in {0, ..., 99} of the proposed move
Reward
Observe unique reward every move

A * log(score/opponent_score) + B * num_islands
Where A increases exponentially from ~0 to 10 and B decreases exponentially from 1 to
~

Reasoning: thought it would encode good behavior at all times in the game, incentivize
early spreading out of territories but later in the game only pay attention to the score, and have
the importance of score matter more and more closer to the end of the game

Training:
Adam optimizer, learning rate 1e-6 (or 1e-4, dont remember)
Memory batch replay size 64

Loss function:
Sparse_softmax_cross_entropy_with_logits
This encourages the model to output logits that are higher for actions that lead to high
rewards

Result:

Crap

Observations:
When I played the blue model, it did indeed play moves that got it islands early on, but
then I would destroy it every game we played, it never made any defensive moves to prevent
itself from getting mini-islanded.

Strategy #2:

NEW REWARD:
Wait until the end of a game to deliver rewards
Then every observation/action during the game gets the reward which is 10 * log(score /
opponent_score)

Train on batch memory replay every 5 moves for both players

Result:

Crap

Strategy #

Deep Blue 1

Red plays randomly

Add regularization to the network

Reward: final score, but weighted by exponential increase over time, so that rewards are more
salient later in the game and less salient earlier in the game

Initial result

Longer result

Crap

Trying similar model with tic tac toe

20,000 iterations :(

Convolutional model getting some results :D

Followed the loss formula for the policy gradient more carefully
Also added learning over epochs to the train loop

Important hyperparameter settings:

Observed behavior

It has a favorite square - the middle! And we can also see, after the middle, a preference for the
corners instead of the non-middle centered squares.

Alternative CNN setup

A problem with policy gradients

Assumes that we are acting with a correspondence between the probabilities pi and the
expected rewards G
But when we include artificial exploration via an epsilon, we no longer update the policy with the
guarantee that the rewards perceived aligned with the probabilities those rewards are aligned
with via the model.
Therefore we should engineer the model itself to bean exploratory agent, instead of layering on
exploration as an external setting that our model cannot participate in - more directly, the
externally-imposed exploration is much harder for our model to learn from, because the train
steps do not discern which actions were taken randomly and which actions were taken
according to model policy.

CURRENT ATTEMPT: ACTOR CRITIC MODEL
Now I think that 100 different possible actions might be too fine-grained of a classification task
for this small of a model. Can we change the way decisions in the game are represented?
Therefore this is more of a model based instead of raw deep-learning algo.

Separate models chained?

Game choices:
-start new island
-expand existing island

Model:
In 1: which # turn is it
In 2: what are the current islands
In 3: the board (10, 10, 4)

Unfortunately I can’t think of a better method than the current heatmap over the 100 spots on
the board.