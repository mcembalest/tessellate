// Game state
let games = [];
let currentGameIndex = -1;
let currentGame = null;
let currentMoveIndex = 0;
let board = [];
let isPlaying = false;
let playInterval = null;
let currentBatchId = null;

// Constants
const RED = 1;
const BLUE = 2;
const EMPTY = 0;
const BLOCKED = 3;
const VISUAL_GRID_SIZE = 5;
const LOGICAL_GRID_SIZE = 10;

// Canvas setup - defer until DOM ready
let canvas, ctx, visualCellSize, logicalCellSize;
let mainSparklineCanvas, mainSparklineCtx;

function initCanvases() {
    canvas = document.getElementById('game-board');
    ctx = canvas.getContext('2d');
    visualCellSize = canvas.width / VISUAL_GRID_SIZE;
    logicalCellSize = canvas.width / LOGICAL_GRID_SIZE;

    mainSparklineCanvas = document.getElementById('main-sparkline');
    mainSparklineCtx = mainSparklineCanvas.getContext('2d');
}
const sparklineCache = new Map();

const colors = {
    [RED]: '#e94560',
    [BLUE]: '#3f72af',
    background: '#37474f',
    gridLines: '#e3e3e3',
    blocked: '#2c3a47'
};

// Initialize empty board
function initBoard() {
    board = Array(LOGICAL_GRID_SIZE).fill(null).map(() => 
        Array(LOGICAL_GRID_SIZE).fill(EMPTY)
    );
}

// Calculate score differences for a game
function calculateScoreDifferences(game) {
    const diffs = [];
    
    // Start with difference of 0 (even game)
    diffs.push(0);
    
    // Calculate difference after each move
    for (let i = 0; i < game.moves.length; i++) {
        let redScore, blueScore;
        
        if (i < game.moves.length - 1) {
            // Get scores from next move's score_before
            const nextMove = game.moves[i + 1];
            redScore = nextMove.score_before[RED] || nextMove.score_before['1'] || 1;
            blueScore = nextMove.score_before[BLUE] || nextMove.score_before['2'] || 1;
        } else {
            // Last move - use final scores
            redScore = game.final_scores.red || 1;
            blueScore = game.final_scores.blue || 1;
        }
        
        // Calculate difference (positive = red winning, negative = blue winning)
        const diff = redScore - blueScore;
        diffs.push(diff);
    }
    
    return diffs;
}

// Draw sparkline
function drawSparkline(canvas, diffs, currentMoveIdx = -1) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(0, 0, width, height);
    
    if (diffs.length < 2) return;
    
    // Find max absolute difference for scaling
    const maxDiff = Math.max(...diffs.map(Math.abs), 10); // Min scale of 10
    const finalDiff = diffs[diffs.length - 1];
    const centerY = height / 2;
    
    // Determine line thickness based on final score difference
    const finalMagnitude = Math.abs(finalDiff);
    let lineWidth = 1;
    if (finalMagnitude < 50) {
        lineWidth = 1; // Close game - thin line
    } else if (finalMagnitude < 200) {
        lineWidth = 2; // Normal victory
    } else {
        lineWidth = 3; // Blowout - thick line
    }
    
    // Draw center line (diff = 0)
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw sparkline with gradient segments
    for (let i = 1; i < diffs.length; i++) {
        const x1 = ((i - 1) / (diffs.length - 1)) * width;
        const x2 = (i / (diffs.length - 1)) * width;
        
        const normalizedDiff1 = diffs[i - 1] / maxDiff;
        const normalizedDiff2 = diffs[i] / maxDiff;
        const y1 = centerY - (normalizedDiff1 * centerY * 0.9);
        const y2 = centerY - (normalizedDiff2 * centerY * 0.9);
        
        // Color based on who's winning at this point
        if (diffs[i] > 0) {
            ctx.strokeStyle = 'rgba(233, 69, 96, 0.9)'; // Red winning
        } else if (diffs[i] < 0) {
            ctx.strokeStyle = 'rgba(63, 114, 175, 0.9)'; // Blue winning
        } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'; // Even
        }
        
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }
    
    // Draw current position marker if specified
    if (currentMoveIdx >= 0 && currentMoveIdx < diffs.length) {
        const x = (currentMoveIdx / (diffs.length - 1)) * width;
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
}

// Draw small sparkline for sidebar
function createSmallSparkline(game) {
    const canvas = document.createElement('canvas');
    canvas.width = 60;
    canvas.height = 20;
    canvas.className = 'sparkline';
    
    const diffs = calculateScoreDifferences(game);
    drawSparkline(canvas, diffs);
    
    return canvas;
}

// Draw the board
function drawBoard() {
    // Recalculate cell sizes
    visualCellSize = canvas.width / VISUAL_GRID_SIZE;
    logicalCellSize = canvas.width / LOGICAL_GRID_SIZE;
    
    // Clear canvas
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw tiles
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const state = board[r][c];
            if (state === RED || state === BLUE) {
                drawTile(r, c, state);
            }
        }
    }

    // Draw grid lines (5x5 visual grid)
    ctx.strokeStyle = colors.gridLines;
    ctx.lineWidth = 1.5;
    for (let i = 0; i <= VISUAL_GRID_SIZE; i++) {
        // Vertical
        ctx.beginPath();
        ctx.moveTo(i * visualCellSize, 0);
        ctx.lineTo(i * visualCellSize, canvas.height);
        ctx.stroke();
        // Horizontal
        ctx.beginPath();
        ctx.moveTo(0, i * visualCellSize);
        ctx.lineTo(canvas.width, i * visualCellSize);
        ctx.stroke();
    }
}

// Draw a triangular tile - EXACT COPY from original
function drawTile(r, c, color) {
    const visualY = Math.floor(r / 2);
    const visualX = Math.floor(c / 2);
    const x0 = visualX * visualCellSize;
    const y0 = visualY * visualCellSize;
    const x1 = (visualX + 1) * visualCellSize;
    const y1 = (visualY + 1) * visualCellSize;
    const tl = { x: x0, y: y0 }; 
    const tr = { x: x1, y: y0 };
    const bl = { x: x0, y: y1 }; 
    const br = { x: x1, y: y1 }; 
    let points = [];
    const isRowEven = r % 2 === 0;
    const isColEven = c % 2 === 0;

    if (isRowEven && isColEven) {
        points = [tl, tr, bl];
    } else if (isRowEven && !isColEven) { 
        points = [tr, br, tl]; 
    } else if (!isRowEven && isColEven) { 
        points = [bl, tl, br];
    } else {
        points = [br, bl, tr];
    }

    ctx.fillStyle = colors[color];
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.lineTo(points[2].x, points[2].y);
    ctx.closePath();
    ctx.fill();
}

// Load games from JSON file
function loadGames(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            games = JSON.parse(e.target.result);
            displayGameList();
            updateStats();
            if (games.length > 0) {
                selectGame(0);
            }
        } catch (error) {
            alert('Error loading games: ' + error.message);
        }
    };
    reader.readAsText(file);
}

// Update statistics
function updateStats() {
    const stats = document.getElementById('stats');
    if (games.length === 0) {
        stats.textContent = 'No games loaded';
        return;
    }

    const redWins = games.filter(g => g.winner === RED).length;
    const blueWins = games.filter(g => g.winner === BLUE).length;
    const ties = games.filter(g => g.winner === null).length;
    
    stats.innerHTML = `
        <strong>${games.length} games</strong><br>
        Red wins: ${redWins} (${(100*redWins/games.length).toFixed(1)}%)<br>
        Blue wins: ${blueWins} (${(100*blueWins/games.length).toFixed(1)}%)<br>
        Ties: ${ties}
    `;
}

// Display game list in sidebar
function displayGameList() {
    const gameList = document.getElementById('game-list');
    gameList.innerHTML = '';

    games.forEach((game, index) => {
        const li = document.createElement('li');
        li.className = 'game-item';
        li.onclick = () => selectGame(index);
        
        // Determine winner and magnitude
        const redScore = game.final_scores.red || 0;
        const blueScore = game.final_scores.blue || 0;
        const scoreDiff = Math.abs(redScore - blueScore);
        
        // Add winner class
        if (game.winner === RED) {
            li.classList.add('red-win');
        } else if (game.winner === BLUE) {
            li.classList.add('blue-win');
        }
        
        // Add magnitude class
        if (scoreDiff < 50) {
            li.classList.add('close');
        } else if (scoreDiff > 200) {
            li.classList.add('blowout');
        }
        
        const winner = game.winner === RED ? 'Red' : 
                        game.winner === BLUE ? 'Blue' : 'Tie';
        
        li.innerHTML = `
            Game ${index + 1} - <strong>${winner}</strong>
            <div class="score">R: ${redScore} | B: ${blueScore}</div>
        `;
        
        // Add sparkline
        const sparkline = createSmallSparkline(game);
        li.appendChild(sparkline);
        
        gameList.appendChild(li);
    });
}

// Select and load a game
function selectGame(index) {
    currentGameIndex = index;
    currentGame = games[index];
    // Start at the end of the game for better UX
    currentMoveIndex = currentGame.moves.length;
    
    // Update UI
    document.querySelectorAll('.game-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
    
    // Setup move slider
    const slider = document.getElementById('move-slider');
    slider.max = currentGame.moves.length;
    slider.value = currentGame.moves.length;
    
    // Update URL with game index (1-indexed for users)
    if (currentBatchId) {
        // Extract simple ID for cleaner URL (just the hash)
        let batchId = currentBatchId;
        if (currentBatchId && currentBatchId.endsWith('.json')) {
            batchId = currentBatchId.replace('.json', '');  // Use full filename without extension
        }
        const newUrl = `${window.location.pathname}?batch=${batchId}&game=${index + 1}`;
        window.history.pushState({}, '', newUrl);
    }
    
    // Reset board and display
    initBoard();
    updateDisplay();
}

// Apply moves up to the current index
function applyMovesToIndex(index) {
    initBoard();
    
    for (let i = 0; i < index; i++) {
        const move = currentGame.moves[i];
        const [r, c] = move.position;
        const player = move.player;
        
        // Place tile
        board[r][c] = player;
        
        // Block adjacent corners
        const c_adj = c + (c % 2 === 0 ? 1 : -1);
        if (c_adj >= 0 && c_adj < LOGICAL_GRID_SIZE) {
            if (board[r][c_adj] === EMPTY) {
                board[r][c_adj] = BLOCKED;
            }
        }
        
        const r_adj = r + (r % 2 === 0 ? 1 : -1);
        if (r_adj >= 0 && r_adj < LOGICAL_GRID_SIZE) {
            if (board[r_adj][c] === EMPTY) {
                board[r_adj][c] = BLOCKED;
            }
        }
    }
}

// Update display
function updateDisplay() {
    if (!currentGame) return;
    
    applyMovesToIndex(currentMoveIndex);
    drawBoard();
    
    // Update scores
    if (currentMoveIndex === 0) {
        document.getElementById('red-score').textContent = '1';
        document.getElementById('blue-score').textContent = '1';
    } else if (currentMoveIndex > 0 && currentMoveIndex <= currentGame.moves.length) {
        const move = currentGame.moves[currentMoveIndex - 1];
        // Get score after this move by looking at next move's score_before or final scores
        if (currentMoveIndex < currentGame.moves.length) {
            const nextMove = currentGame.moves[currentMoveIndex];
            document.getElementById('red-score').textContent = nextMove.score_before[RED] || nextMove.score_before['1'];
            document.getElementById('blue-score').textContent = nextMove.score_before[BLUE] || nextMove.score_before['2'];
        } else {
            document.getElementById('red-score').textContent = currentGame.final_scores.red;
            document.getElementById('blue-score').textContent = currentGame.final_scores.blue;
        }
    }
    
    // Update move info
    document.getElementById('current-move').textContent = currentMoveIndex;
    document.getElementById('total-moves').textContent = currentGame.moves.length;
    
    // Current player
    const currentPlayer = currentMoveIndex < currentGame.moves.length ? 
        (currentGame.moves[currentMoveIndex].player === RED ? 'Red' : 'Blue') : 
        'Game Over';
    document.getElementById('current-player').textContent = currentPlayer === 'Game Over' ? currentPlayer : `${currentPlayer}'s turn`;
    
    // Update slider
    document.getElementById('move-slider').value = currentMoveIndex;
    
    // Update button states
    document.getElementById('first-btn').disabled = currentMoveIndex === 0;
    document.getElementById('prev-btn').disabled = currentMoveIndex === 0;
    document.getElementById('next-btn').disabled = currentMoveIndex >= currentGame.moves.length;
    document.getElementById('last-btn').disabled = currentMoveIndex >= currentGame.moves.length;
    
    // Update main sparkline
    const diffs = calculateScoreDifferences(currentGame);
    drawSparkline(mainSparklineCanvas, diffs, currentMoveIndex);
}

// Navigation functions
function firstMove() {
    currentMoveIndex = 0;
    updateDisplay();
}

function prevMove() {
    if (currentMoveIndex > 0) {
        currentMoveIndex--;
        updateDisplay();
    }
}

function nextMove() {
    if (currentGame && currentMoveIndex < currentGame.moves.length) {
        currentMoveIndex++;
        updateDisplay();
    }
}

function lastMove() {
    if (currentGame) {
        currentMoveIndex = currentGame.moves.length;
        updateDisplay();
    }
}

function togglePlay() {
    if (isPlaying) {
        stopPlay();
    } else {
        startPlay();
    }
}

function startPlay() {
    if (!currentGame || currentMoveIndex >= currentGame.moves.length) {
        currentMoveIndex = 0;
    }
    
    isPlaying = true;
    document.getElementById('play-btn').textContent = '⏸️';
    
    playInterval = setInterval(() => {
        if (currentMoveIndex >= currentGame.moves.length) {
            stopPlay();
        } else {
            nextMove();
        }
    }, 500); // 500ms per move
}

function stopPlay() {
    isPlaying = false;
    document.getElementById('play-btn').textContent = '▶️';
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

// Event listeners
document.getElementById('file-input').addEventListener('change', loadGames);
document.getElementById('first-btn').addEventListener('click', firstMove);
document.getElementById('prev-btn').addEventListener('click', prevMove);
document.getElementById('next-btn').addEventListener('click', nextMove);
document.getElementById('last-btn').addEventListener('click', lastMove);
document.getElementById('play-btn').addEventListener('click', togglePlay);

// Canvas event listeners will be set up in setupEventListeners()
/* mainSparklineCanvas.addEventListener('click', (e) => {
    if (!currentGame) return;
    
    const rect = mainSparklineCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / mainSparklineCanvas.width;
    const targetMove = Math.round(ratio * currentGame.moves.length);
    
    currentMoveIndex = Math.max(0, Math.min(targetMove, currentGame.moves.length));
    updateDisplay();
});

// Main sparkline hover tooltip
let tooltip = null;
mainSparklineCanvas.addEventListener('mousemove', (e) => {
    if (!currentGame) return;
    
    const rect = mainSparklineCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / mainSparklineCanvas.width;
    const targetMove = Math.round(ratio * currentGame.moves.length);
    
    // Create or update tooltip
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'sparkline-tooltip';
        document.body.appendChild(tooltip);
    }
    
    // Get scores at this move
    let redScore = 1, blueScore = 1;
    if (targetMove > 0 && targetMove <= currentGame.moves.length) {
        if (targetMove < currentGame.moves.length) {
            const nextMove = currentGame.moves[targetMove];
            redScore = nextMove.score_before[RED] || nextMove.score_before['1'] || 1;
            blueScore = nextMove.score_before[BLUE] || nextMove.score_before['2'] || 1;
        } else {
            redScore = currentGame.final_scores.red || 1;
            blueScore = currentGame.final_scores.blue || 1;
        }
    }
    
    tooltip.innerHTML = `Move ${targetMove}<br>R: ${redScore} | B: ${blueScore}`;
    tooltip.style.left = e.clientX + 10 + 'px';
    tooltip.style.top = e.clientY - 30 + 'px';
    tooltip.style.display = 'block';
});

}) */

// All canvas event listeners moved to setupEventListeners()

// // Batch selector change handler
// document.getElementById('batch-selector').addEventListener('change', async (e) => {
//     const batchFile = e.target.value;
//     if (batchFile) {
//         const loaded = await loadBatchById(batchFile);
//         if (loaded && games.length > 0) {
//             selectGame(0);
//             // Use full filename for URL (without extension)
//             const batchId = batchFile.replace('.json', '');
//             // Update URL without reloading (game=1 for first game)
//             const newUrl = `${window.location.pathname}?batch=${batchId}&game=1`;
//             window.history.pushState({}, '', newUrl);
//         }
//     }
// });

document.getElementById('move-slider').addEventListener('input', (e) => {
    currentMoveIndex = parseInt(e.target.value);
    updateDisplay();
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (!currentGame) return;
    
    switch(e.key) {
        case 'ArrowLeft':
            prevMove();
            break;
        case 'ArrowRight':
            nextMove();
            break;
        case ' ':
            e.preventDefault();
            togglePlay();
            break;
        case 'Home':
            firstMove();
            break;
        case 'End':
            lastMove();
            break;
    }
});

// Parse URL parameters
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    return {
        batch: params.get('batch'),
        // Convert 1-indexed URL param to 0-indexed array index
        game: params.get('game') ? parseInt(params.get('game')) - 1 : null
    };
}

// // Load available game files
// async function loadGameDirectory() {
//     // For GitHub Pages, we can't scan directories
//     // So we'll just hide the batch selector and rely on default games
//     const selector = document.getElementById('batch-selector');
    
//     // Check if we're on localhost (where directory scanning might work)
//     const isLocalhost = window.location.hostname === 'localhost' || 
//                         window.location.hostname === '127.0.0.1';
    
//     if (isLocalhost) {
//         try {
//             // Try to fetch the game_data directory listing
//             const response = await fetch('game_data/');
//             if (response.ok) {
//                 const text = await response.text();
//                 // Parse HTML directory listing for .json files
//                 const matches = text.match(/href="([^"]+\.json)"/g);
//                 if (matches) {
//                     const files = matches.map(m => m.match(/href="([^"]+)"/)[1]);
//                     const batchFiles = files.filter(f => f.startsWith('batch_'));
//                     console.log(`Found ${batchFiles.length} batch files`);
//                     populateBatchSelectorFromFiles(batchFiles);
//                     selector.style.display = 'block';
//                     return true;
//                 }
//             }
//         } catch (error) {
//             console.log('Could not scan directory:', error);
//         }
//     }
    
//     // For GitHub Pages or if scanning fails, hide the selector
//     selector.style.display = 'none';
//     return true;
// }

// Populate batch selector from file list
function populateBatchSelectorFromFiles(batchFiles) {
    const selector = document.getElementById('batch-selector');
    selector.innerHTML = '<option value="">Select Batch...</option>';
    
    if (batchFiles && batchFiles.length > 0) {
        batchFiles.forEach(filename => {
            // Extract batch ID from filename
            // Handle both old format (batch_20250815_130917_cc23eaae.json)
            // and new format (batch_20250815_142852_2ff4c4_001_bed6e594.json)
            let batchId;
            const parts = filename.replace('.json', '').split('_');
            
            if (parts.length >= 4) {
                // Try to extract the hash ID (last 8-char hex string)
                batchId = parts[parts.length - 1];
                if (batchId.length !== 8) {
                    // For new format, the hash is the last part
                    batchId = parts[parts.length - 1];
                }
            }
            
            const option = document.createElement('option');
            option.value = filename;  // Store full filename
            option.textContent = filename.replace('.json', '').replace('batch_', '');
            selector.appendChild(option);
        });
    }
    
    // Set current batch as selected
    if (currentBatchId) {
        selector.value = currentBatchId;
    }
}

// Load specific batch by ID or filename
async function loadBatchById(batchIdOrFile) {
    try {
        // Determine the URL to fetch
        let url;
        if (batchIdOrFile.endsWith('.json')) {
            // Full filename provided
            url = `game_data/${batchIdOrFile}`;
            currentBatchId = batchIdOrFile;
        } else {
            // Just batch ID provided - try to find matching file
            // This handles URLs with just the hash ID
            url = `game_data/batch_*${batchIdOrFile}.json`;
            // Try direct filename first
            const possibleFiles = [
                `game_data/${batchIdOrFile}.json`
            ];
            // Try each possible file
            for (const file of possibleFiles) {
                try {
                    const response = await fetch(file);
                    if (response.ok) {
                        games = await response.json();
                        currentBatchId = file.split('/')[1];  // Get filename
                        console.log(`Loaded batch ${batchIdOrFile} with ${games.length} games`);
                        displayGameList();
                        updateStats();
                        return true;
                    }
                } catch (e) {}
            }
            // If not found by ID, treat as filename
            url = `game_data/${batchIdOrFile}`;
            currentBatchId = batchIdOrFile;
        }
        
        const response = await fetch(url);
        if (response.ok) {
            games = await response.json();
            console.log(`Loaded batch ${batchIdOrFile} with ${games.length} games`);
            displayGameList();
            updateStats();
            return true;
        }
    } catch (error) {
        console.log(`Error loading batch ${batchIdOrFile}:`, error);
    }
    return false;
}

// Initialize app with URL params or defaults
async function initializeApp() {
    const urlParams = getUrlParams();
    
    // Load game directory first
    // await loadGameDirectory();
    
    // Try to load specific batch if provided
    if (urlParams.batch) {
        const loaded = await loadBatchById(urlParams.batch);
        if (loaded && urlParams.game !== null) {
            // Jump to specific game (urlParams.game is already 0-indexed from getUrlParams)
            selectGame(urlParams.game);
        } else if (loaded) {
            // Select first game
            if (games.length > 0) selectGame(0);
        }
    } else {
        // Default: load hardcoded batch file  
        const defaultBatch = 'batch_20250815_130917_cc23eaae';
        console.log(`Loading default batch: ${defaultBatch}`);
        const loaded = await loadBatchById(defaultBatch);
        if (!loaded || games.length === 0) {
            console.log('Default batch not found, loading sample games');
            await loadSampleGames();
        } else {
            if (games.length > 0) selectGame(0);
        }
    }
}

// Fallback for when no games can be loaded
async function loadSampleGames() {
    const possibleUrls = [
        'game_data/batch_20250815_130917_cc23eaae.json',
        'game_data/batch_20250815_130956_20559b7d.json',
        'game_data/batch_20250815_131035_69f49367.json'
    ];
    
    // Try each possible URL until one works
    for (const url of possibleUrls) {
        try {
            const response = await fetch(url);
            if (response.ok) {
                games = await response.json();
                console.log(`Loaded ${games.length} games from ${url}`);
                displayGameList();
                updateStats();
                if (games.length > 0) {
                    selectGame(0);
                }
                return;
            }
        } catch (error) {
            console.log(`Could not load from ${url}:`, error);
        }
    }
    
    // If no files could be loaded (likely due to CORS when opening via file://)
    console.log('Could not auto-load games.');
    games = [];
}

function setupEventListeners() {
    // Canvas-dependent event listeners
    mainSparklineCanvas.addEventListener('click', (e) => {
        if (!currentGame) return;
        
        const rect = mainSparklineCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const ratio = x / mainSparklineCanvas.width;
        const targetMove = Math.floor(ratio * currentGame.moves.length);
        
        selectMove(Math.min(targetMove, currentGame.moves.length));
    });

    mainSparklineCanvas.addEventListener('mousemove', (e) => {
        if (!currentGame) return;
        
        const rect = mainSparklineCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const ratio = x / mainSparklineCanvas.width;
        const targetMove = Math.floor(ratio * currentGame.moves.length);
        
        // Show tooltip
        const tooltip = document.getElementById('tooltip');
        
        // Get scores at this move
        let redScore = 1, blueScore = 1;
        if (targetMove > 0 && targetMove <= currentGame.moves.length) {
            if (targetMove < currentGame.moves.length) {
                const nextMove = currentGame.moves[targetMove];
                redScore = nextMove.score_before[RED] || nextMove.score_before['1'] || 1;
                blueScore = nextMove.score_before[BLUE] || nextMove.score_before['2'] || 1;
            } else {
                redScore = currentGame.final_scores.red || 1;
                blueScore = currentGame.final_scores.blue || 1;
            }
        }
        
        tooltip.innerHTML = `Move ${targetMove}<br>R: ${redScore} | B: ${blueScore}`;
        tooltip.style.left = e.clientX + 10 + 'px';
        tooltip.style.top = e.clientY - 30 + 'px';
        tooltip.style.display = 'block';
    });

    mainSparklineCanvas.addEventListener('mouseleave', () => {
        const tooltip = document.getElementById('tooltip');
        tooltip.style.display = 'none';
    });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    initCanvases();
    setupEventListeners();
    initBoard();
    drawBoard();
    await initializeApp();
});