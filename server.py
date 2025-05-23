import asyncio
import websockets
import json
import uuid

# Game constants
RED = 0
BLUE = 1
EMPTY = 2
OUT_OF_PLAY = 3  # Not strictly needed for server turn validation but good for consistency
VISUAL_GRID_SIZE = 5
LOGICAL_GRID_SIZE = VISUAL_GRID_SIZE * 2
TOTAL_TILES = VISUAL_GRID_SIZE * VISUAL_GRID_SIZE * 2

# Global dictionary to store active game sessions
active_sessions = {}

async def handler(websocket, path):
    """
    Manages individual client connections and processes incoming messages.
    """
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "create_game":
                session_id = uuid.uuid4().hex[:6]
                active_sessions[session_id] = {
                    "board": [[EMPTY for _ in range(LOGICAL_GRID_SIZE)] for _ in range(LOGICAL_GRID_SIZE)],
                    "current_turn": RED,
                    "scores": {RED: 0, BLUE: 0}, # Initial scores
                    "players": [None, None], # Index 0 for RED, 1 for BLUE
                    "game_over_flag": False,
                    "connected_clients": 0 # Keep track of connected clients in this session
                }
                # Store Player 1 (RED)
                active_sessions[session_id]["players"][RED] = websocket
                active_sessions[session_id]["connected_clients"] += 1

                await websocket.send(json.dumps({
                    "type": "game_created",
                    "session_id": session_id,
                    "player_id": RED
                }))
                print(f"Game session {session_id} created by {websocket.remote_address}. Player RED assigned.")

            elif message_type == "join_game":
                session_id = data.get("session_id")
                session = active_sessions.get(session_id)

                if session and session["players"][BLUE] is None:
                    session["players"][BLUE] = websocket
                    session["connected_clients"] += 1

                    await websocket.send(json.dumps({
                        "type": "player_joined",
                        "session_id": session_id,
                        "player_id": BLUE
                    }))
                    print(f"Player BLUE joined session {session_id} ({websocket.remote_address})")

                    # Notify Player RED (index 0) that Player BLUE has joined
                    if session["players"][RED]:
                         await session["players"][RED].send(json.dumps({
                            "type": "opponent_joined",
                            "session_id": session_id
                        }))

                    # Send initial game update to both players
                    initial_game_update = {
                        "type": "game_update",
                        "board": session["board"],
                        "current_turn": session["current_turn"],
                        "scores": session["scores"],
                        "game_over_flag": session["game_over_flag"]
                    }
                    await broadcast(session_id, initial_game_update)
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Session not found or full"
                    }))
                    print(f"Failed attempt to join session {session_id} by {websocket.remote_address}: Session not found or full.")

            elif message_type == "make_move":
                session_id = data.get("session_id")
                player_id = data.get("player_id")
                move = data.get("move") # Expected to be a dict like {"r": r, "c": c}
                session = active_sessions.get(session_id)

                if not session:
                    await websocket.send(json.dumps({"type": "error", "message": "Session not found"}))
                    return
                if session["game_over_flag"]:
                    await websocket.send(json.dumps({"type": "error", "message": "Game is over"}))
                    return
                if session["current_turn"] != player_id:
                    await websocket.send(json.dumps({"type": "error", "message": "Not your turn"}))
                    return
                # Optional: Add more detailed move validation (e.g., cell empty, within bounds)
                # For now, we assume client sends valid cell coordinates
                r, c = move.get("r"), move.get("c")
                if not (0 <= r < LOGICAL_GRID_SIZE and 0 <= c < LOGICAL_GRID_SIZE and session["board"][r][c] == EMPTY):
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid move"}))
                    return

                # Apply the move
                session["board"][r][c] = player_id
                # Simple score update: count tiles for each player
                session["scores"][RED] = sum(row.count(RED) for row in session["board"])
                session["scores"][BLUE] = sum(row.count(BLUE) for row in session["board"])

                # Switch turn
                session["current_turn"] = BLUE if player_id == RED else RED

                # Check for game over (e.g., board full)
                # A more sophisticated check would involve checking for valid moves left
                if (session["scores"][RED] + session["scores"][BLUE]) >= TOTAL_TILES:
                    session["game_over_flag"] = True
                    # Determine winner (can be more complex)
                    winner = RED if session["scores"][RED] > session["scores"][BLUE] else BLUE if session["scores"][BLUE] > session["scores"][RED] else "draw"
                    print(f"Game over in session {session_id}. Winner: {winner}")
                    await broadcast(session_id, {"type": "game_over", "winner": winner, "scores": session["scores"]})


                # Broadcast game update
                game_update_message = {
                    "type": "game_update",
                    "board": session["board"],
                    "current_turn": session["current_turn"],
                    "scores": session["scores"],
                    "game_over_flag": session["game_over_flag"]
                }
                await broadcast(session_id, game_update_message)
                print(f"Move made in session {session_id} by player {player_id}. Cell: ({r},{c})")

            else:
                print(f"Unknown message type: {message_type}")
                await websocket.send(json.dumps({"type": "error", "message": "Unknown message type"}))

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client disconnected: {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection error: {websocket.remote_address} - {e}")
    finally:
        # Handle client disconnection
        disconnected_session_id = None
        disconnected_player_id = None
        for sid, session_data in active_sessions.items():
            for i, player_ws in enumerate(session_data["players"]):
                if player_ws == websocket:
                    disconnected_session_id = sid
                    disconnected_player_id = i
                    session_data["players"][i] = None # Mark player as disconnected
                    session_data["connected_clients"] -= 1
                    print(f"Player {disconnected_player_id} disconnected from session {disconnected_session_id}.")
                    break
            if disconnected_session_id:
                break
        
        if disconnected_session_id:
            session = active_sessions[disconnected_session_id]
            remaining_player_id = 1 - disconnected_player_id
            remaining_player_socket = session["players"][remaining_player_id]

            if remaining_player_socket and not session["game_over_flag"]:
                # Game was ongoing, remaining player wins by opponent's disconnection
                session["game_over_flag"] = True
                # Scores can be left as is, or updated if desired.
                # For example, ensure the winner has more points if that matters for client display.
                # Let's assume scores are as they were when disconnection happened for now.
                
                winner_message = f"Player {PLAYER_COLOR_MAP.get(disconnected_player_id, 'Unknown')} disconnected. Player {PLAYER_COLOR_MAP.get(remaining_player_id, 'Unknown')} wins!"
                print(f"Session {disconnected_session_id}: {winner_message}")

                game_over_data = {
                    "type": "game_over",
                    "winner": remaining_player_id,
                    "scores": session["scores"],
                    "message": winner_message # Custom message for this scenario
                }
                try:
                    await remaining_player_socket.send(json.dumps(game_over_data))
                except websockets.exceptions.ConnectionClosed:
                    print(f"Failed to send game_over to remaining player in session {disconnected_session_id} as they also disconnected.")
            
            elif remaining_player_socket and session["game_over_flag"]:
                 # Game was already over, just inform about disconnection if needed (optional)
                 # This case is less critical as game already concluded.
                 # The existing "opponent_disconnected" message might suffice if sent before this block.
                 # For now, we focus on the win-by-disconnection scenario.
                 pass


            # If both players are disconnected (or remaining player was already None), clean up the session
            if session["connected_clients"] == 0:
                print(f"Session {disconnected_session_id} has no connected clients. Removing session.")
                del active_sessions[disconnected_session_id]


async def broadcast(session_id, message):
    """
    Sends a JSON message to both connected players in a given session.
    """
    session = active_sessions.get(session_id)
    if session:
        players = session.get("players", [])
        for player_ws in players:
            if player_ws: # Check if player_ws is not None
                try:
                    await player_ws.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    # Handle if a player in the session has disconnected
                    print(f"Attempted to send to a disconnected client in session {session_id}")


async def main():
    """
    Starts the WebSocket server.
    """
    host = 'localhost'
    port = 8765
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
