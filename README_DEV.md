# Tessellate Development Guide

## Quick Start (Development)

To run Tessellate locally with full functionality:

```bash
# Start the development server
python3 start_server.py

# Or using Python's built-in server
python3 -m http.server 8000
```

Then open: http://localhost:8000/

## Why a Local Server?

When opening HTML files directly (file:// protocol), browsers block loading JSON files due to CORS security policies. Running a local server solves this and makes development match production behavior.

## File Structure

- `index.html` - Main game (Player vs Player)
- `browser.html` - Game browser (auto-loads 1000 games)
- `game-pvp.js` - Two-player game logic
- `tessellate.css` - Shared styles
- `random_games_1000.json` - Pre-generated games for browsing

## Navigation

1. **Home Page** (index.html) - Play Tessellate
2. **Game Browser** (browser.html) - Browse and replay 1000 games
   - Auto-loads games when served via HTTP
   - Shows move-by-move replay
   - Keyboard navigation (← → arrows)

## Production Deployment

When deploying to production, update the `PRODUCTION_DATA_URL` in browser.html to point to your API endpoint:

```javascript
const PRODUCTION_DATA_URL = 'https://your-api.com/games/1000';
```

The browser will automatically use this URL when not on localhost.