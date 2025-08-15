#!/usr/bin/env python3
"""
Fast single-threaded game generation with unique session IDs
Prevents file collisions when running multiple generators
"""

import json
import time
import sys
import os
import hashlib
import random
from datetime import datetime
from pathlib import Path
from agents import RandomAgent, play_game

def generate_session_id():
    """Generate unique session ID for this generation run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.sha256(f"{time.time()}_{random.random()}".encode()).hexdigest()[:6]
    return f"{timestamp}_{random_hash}"

def update_registry(session_id, batch_info):
    """Update the game directory registry atomically"""
    registry_path = Path("game_data/game_directory.json")
    
    # Load existing registry or create new one
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_games": 0,
            "batches": [],
            "complete_files": []
        }
    
    # Add new batch info
    registry["batches"].append(batch_info)
    registry["total_games"] += batch_info["games"]
    
    # Write atomically (write to temp, then rename)
    temp_path = registry_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(registry, f, indent=2)
    temp_path.rename(registry_path)
    
    return registry["total_games"]

def generate_games_fast(n_games, save_every=10000):
    """Generate games with unique session ID to prevent collisions"""
    
    # Generate unique session ID
    session_id = generate_session_id()
    
    print(f"=== Fast Game Generation ===")
    print(f"Session ID: {session_id}")
    print(f"Target: {n_games:,} games")
    print(f"Saving every: {save_every:,} games")
    print("="*30)
    
    all_games = []
    start_time = time.time()
    last_save = 0
    batch_num = 0
    
    for i in range(n_games):
        # Create agents and play game
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        game_record = play_game(agent1, agent2, verbose=False)
        all_games.append(game_record)
        
        # Progress update every 1000 games
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_games - i - 1) / rate
            
            print(f"[{elapsed/60:.1f}m] {i+1:,}/{n_games:,} games "
                  f"({100*(i+1)/n_games:.1f}%) - "
                  f"{rate:.1f} games/sec - ETA: {eta/60:.1f}m")
            
            # Write progress file
            with open('progress.txt', 'w') as f:
                f.write(f"Games: {i+1:,} / {n_games:,}\n")
                f.write(f"Progress: {100*(i+1)/n_games:.1f}%\n")
                f.write(f"Rate: {rate:.1f} games/sec\n")
                f.write(f"Elapsed: {elapsed/60:.1f} minutes\n")
                f.write(f"ETA: {eta/60:.1f} minutes\n")
        
        # Save batch every save_every games
        if (i + 1) % save_every == 0:
            batch_num += 1
            batch_games = all_games[last_save:i+1]
            
            # Create unique filename with session ID
            batch_hash = hashlib.sha256(f"{session_id}_{batch_num}".encode()).hexdigest()[:8]
            filename = f"game_data/batch_{session_id}_{batch_num:03d}_{batch_hash}.json"
            
            # Save batch
            with open(filename, 'w') as f:
                json.dump(batch_games, f)
            
            # Update registry
            batch_info = {
                "id": batch_hash,
                "session": session_id,
                "batch_number": batch_num,
                "file": os.path.basename(filename),
                "games": len(batch_games),
                "created": datetime.now().isoformat()
            }
            total_registered = update_registry(session_id, batch_info)
            
            print(f"  → Saved batch {batch_num} ({save_every:,} games) - Total: {total_registered:,}")
            last_save = i + 1
    
    # Save remaining games
    if last_save < n_games:
        batch_num += 1
        remaining_games = all_games[last_save:]
        remaining = len(remaining_games)
        
        # Create unique filename with session ID
        batch_hash = hashlib.sha256(f"{session_id}_{batch_num}".encode()).hexdigest()[:8]
        filename = f"game_data/batch_{session_id}_{batch_num:03d}_{batch_hash}.json"
        
        # Save batch
        with open(filename, 'w') as f:
            json.dump(remaining_games, f)
        
        # Update registry
        batch_info = {
            "id": batch_hash,
            "session": session_id,
            "batch_number": batch_num,
            "file": os.path.basename(filename),
            "games": remaining,
            "created": datetime.now().isoformat()
        }
        total_registered = update_registry(session_id, batch_info)
        
        print(f"  → Saved final batch {batch_num} ({remaining:,} games) - Total: {total_registered:,}")
    
    # Save complete dataset with session ID
    elapsed = time.time() - start_time
    print(f"\nSaving complete dataset...")
    complete_filename = f"game_data/all_games_{session_id}_{n_games}.json"
    with open(complete_filename, 'w') as f:
        json.dump(all_games, f)
    
    print(f"\n=== Complete ===")
    print(f"Generated: {n_games:,} games")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Rate: {n_games/elapsed:.1f} games/sec")
    
    return all_games

if __name__ == "__main__":
    import os
    os.makedirs("game_data", exist_ok=True)
    
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    generate_games_fast(n_games)