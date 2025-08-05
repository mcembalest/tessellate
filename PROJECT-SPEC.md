# Tessellate Project Specification

These are the rules of Tessellate. They explain the process of taking turns, scoring, and winning. They apply to any board composed of squares. Note that a player's score may not always increase.

1. A player's turn consists of placing a tile with their color on the board.

2. A tile fills one half of a square with a right triangle.

3. Tiles may not be placed on top of each other.

4. Tiles of the same color that border each other form an island.

5. The size of an island is the number of tiles in it.

6. A player's score is the product of all their islands (the multiplication of their islands' sizes).

7. The game ends when all the squares are filled.

8. The player with the highest score at the end of the game wins.

## Core Requirements

### 1. Game Persistence
- **WHAT**: Save game trajectories (both partial and complete)
- **WHEN**: Locally after each move, online when published/completed
- **DATA NEEDED**: 
  - Full sequence of moves (player, position)
  - Score progression after each move
  - Island size distribution after each move (largest first)
- **PURPOSE**: Study score dynamics, enable game resumption, analyze island merges
- **NOT**: Move timestamps, player profiles, pre-computed analysis

### 2. Game Sharing
- **WHAT**: 7-character alphanumeric game IDs (case-sensitive)
- **HOW**: Direct link with mode parameter (?mode=spectate)
- **BROWSING**: Recent games list with search by ID
- **PUBLISHING**: Bulk publish with validation (50 moves, 25 per color, all positions used)
- **GENERATION**: Always random, no custom IDs
- **NOT**: User accounts, game passwords, private games

### 3. Multiplayer  
- **WHAT**: Turn-based play via shared link
- **HOW**: Polling every 2 seconds for opponent moves
- **TURN ORDER**: Red plays first, then Blue alternates
- **TIMEOUT**: Human games: 24 hours, Bot games: 1 hour
- **VALIDATION**: Prevent duplicate and invalid move submissions
- **NOT**: Real-time websockets, chat, notifications

### 4. Bot Support
- **WHAT**: Bots play games move-by-move (including self-play)
- **SPEED**: Maximum throughput for RL training (thousands of games)
- **INTERFACE**: Simple HTTP API with curl/requests examples
- **BATCH**: Create multiple games in one request for parallel play
- **TIMEOUT**: 1 hour per game
- **NOT**: Pre-computed games, bot profiles, rankings, move explanations

### 5. Analytics Export
- **WHAT**: Bulk export all games (complete and partial)
- **DATA INCLUDED**: 
  - Move-by-move trajectories
  - Score evolution throughout each game
  - Island size distributions at each step (largest first)
  - Tile-to-island mappings
- **PURPOSE**: Enable external analysis of game dynamics and incomplete strategies
- **NOT**: Real-time analytics, dashboards, visualizations

## Technical Constraints

### Frontend
- Single `index.html` file on GitHub Pages  
- Three play modes: spectating (replay), playing (with confirm), speed-playing (instant)
- Local-first: games save locally until explicitly published online
- Game browser with board thumbnail previews
- Warning dialog when abandoning mid-game
- No build process, no framework

### Development Mode
- Local bot-vs-bot games for testing
- Ephemeral in-memory storage (no database required)
- Random move selection for debugging
- Fast game completion without network calls
- Same game logic as production

### Backend  
- Vercel serverless functions
- PostgreSQL database
- REST API (no GraphQL)
- Polling (no WebSockets)


## Required Capabilities

### Game Operations
- Create new games (human or bot)
- Retrieve game state and history
- Submit moves with validation
- Track score and island evolution
- Browse all games via web interface

### Data Access
- Export all games as JSON lines stream
- API to list all games and get individual games by ID
- Include full trajectory data
- Preserve all games indefinitely
- Client-side generated 100x100px board thumbnails
- Browse by most recent with search by game ID

## What We DON'T Need
- User authentication
- Game replays with animation  
- Move validation explanations (just accept/reject)
- Separate tables for moves/analytics
- Admin dashboard
- Custom game IDs
- Move timestamps
- Connection status indicators

## Success Criteria
1. Can play a full game locally
2. Can run instant bot-vs-bot games in development mode
3. Can share link and play with friend
4. Multiplayer games resume correctly after browser refresh
5. Bots can play 1000+ concurrent games (including self-play)
6. Can export all games (partial and complete) with full trajectories
7. Games persist with move history, scores, and tile-to-island mappings
8. Can analyze how scores and islands evolve during gameplay