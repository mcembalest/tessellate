// Fully client-side Tessellategame logic

const RED = 0;
const BLUE = 1;
const EMPTY = 2;
const OUT_OF_PLAY = 3;
const VISUAL_GRID_SIZE = 5;
const LOGICAL_GRID_SIZE = VISUAL_GRID_SIZE * 2; 
const TOTAL_TILES = VISUAL_GRID_SIZE * VISUAL_GRID_SIZE * 2; 

let showPreview = true;
let visualCellSize = 0; 
let logicalCellSize = 0;
let board = [];
let currentTurn = RED;
let scores = { [RED]: 1, [BLUE]: 1 };
let gameOver = false;
let hoverPosition = null; 
let placedTilesCount = 0;

let canvas, ctx;
const colors = {
    [RED]: { normal: '#e94560', preview: '#ff96a6' },
    [BLUE]: { normal: '#3f72af', preview: '#94b8ff' },
    background: '#37474f',
    gridLines: '#e3e3e3',
};

function initializeBoard() {
    board = Array(LOGICAL_GRID_SIZE).fill(null).map(() =>
        Array(LOGICAL_GRID_SIZE).fill(EMPTY)
    );
    currentTurn = RED;
    scores = { [RED]: 1, [BLUE]: 1 };
    hoverPosition = null;
    gameOver = false;
    placedTilesCount = 0;
    console.log(`Board Initialized (${LOGICAL_GRID_SIZE}x${LOGICAL_GRID_SIZE} Logical Grid)`);
}

function setupCanvas() {
    canvas = document.getElementById('game-board');
    ctx = canvas.getContext('2d');
    
    const container = document.getElementById('game-container');
    const containerSize = container.offsetWidth;
    canvas.width = containerSize;
    canvas.height = containerSize;
    visualCellSize = canvas.width / VISUAL_GRID_SIZE;
    logicalCellSize = visualCellSize / 2;
    console.log(`Canvas Setup: Size=${canvas.width}x${canvas.height}, VisualCell=${visualCellSize}, LogicalCell=${logicalCellSize}`);
    drawBoard();
}

function drawBoard() {
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const tileState = board[r][c];
            if (tileState === RED || tileState === BLUE) {
                drawTile(r, c, tileState, false);
            }
        }
    }
    
    ctx.strokeStyle = colors.gridLines;
    ctx.lineWidth = 1.5;
    for (let i = 0; i <= VISUAL_GRID_SIZE; i++) {
        // vertical grid lines
        ctx.beginPath();
        ctx.moveTo(i * visualCellSize, 0);
        ctx.lineTo(i * visualCellSize, canvas.height);
        ctx.stroke();
        // horizontal grid lines
        ctx.beginPath();
        ctx.moveTo(0, i * visualCellSize);
        ctx.lineTo(canvas.width, i * visualCellSize);
        ctx.stroke();
    }
    
    if (showPreview && hoverPosition && !gameOver && isPlayable(hoverPosition.r, hoverPosition.c)) {
        drawTile(hoverPosition.r, hoverPosition.c, currentTurn, true);
    }
}

function drawTile(r, c, color, isPreview) {
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
    
    ctx.fillStyle = isPreview ? colors[color].preview : colors[color].normal;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    ctx.lineTo(points[1].x, points[1].y);
    ctx.lineTo(points[2].x, points[2].y);
    ctx.closePath();
    ctx.fill();
}

function isValidCoord(r, c) {
    return r >= 0 && r < LOGICAL_GRID_SIZE && c >= 0 && c < LOGICAL_GRID_SIZE;
}

function isPlayable(r, c) {
    return isValidCoord(r, c) && board[r][c] === EMPTY;
}

function addTile(r, c) {
    if (!isPlayable(r, c)) {
        console.error(`Internal Error: Attempted to play invalid move at logical corner (${r}, ${c})`);
        return false;
    }
    
    console.log(`Adding ${currentTurn === RED ? 'RED' : 'BLUE'} tile at corner (${r}, ${c})`);
    board[r][c] = currentTurn;
    placedTilesCount++;
    
    // block adjacent corners
    const c_adj = c + (c % 2 === 0 ? 1 : -1);
    if (isValidCoord(r, c_adj)) {
        if (board[r][c_adj] === EMPTY) board[r][c_adj] = OUT_OF_PLAY;
    }
    
    const r_adj = r + (r % 2 === 0 ? 1 : -1);
    if (isValidCoord(r_adj, c)) {
        if (board[r_adj][c] === EMPTY) board[r_adj][c] = OUT_OF_PLAY;
    }

    return true;
}

function calculateScore() {
    scores = { [RED]: 1, [BLUE]: 1 };
    const visited = Array(LOGICAL_GRID_SIZE).fill(null).map(() =>
        Array(LOGICAL_GRID_SIZE).fill(false)
    );
    
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const color = board[r][c];
            if ((color === RED || color === BLUE) && !visited[r][c]) {
                const regionSize = dfs(r, c, color, visited);
                if (regionSize > 0) {
                    scores[color] *= regionSize;
                    console.log(`Found ${color === RED ? 'RED' : 'BLUE'} region size ${regionSize} starting at (${r},${c}). New score prod: ${scores[color]}`);
                }
            }
        }
    }
    console.log(`Final Calculated Scores: RED=${scores[RED]}, BLUE=${scores[BLUE]}`);
}

function dfs(startR, startC, color, visited) {
    let size = 0;
    const stack = [[startR, startC]];
    while (stack.length > 0) {
        const [r, c] = stack.pop();
        if (isValidCoord(r, c) && board[r][c] === color && !visited[r][c]) {
            visited[r][c] = true;
            size++;
            const neighbors = getNeighbors(r, c);
            for (const [nr, nc] of neighbors) {
                if (isValidCoord(nr, nc) && board[nr][nc] === color && !visited[nr][nc]) {
                    stack.push([nr, nc]);
                }
            }
        }
    }
    return size;
}

function getNeighbors(r, c) {
    const neighbors = [];
    const pow_neg1_r = (r % 2 === 0 ? 1 : -1);
    const pow_neg1_c = (c % 2 === 0 ? 1 : -1);
    const pow_neg1_r_plus_1 = ((r + 1) % 2 === 0 ? 1 : -1);
    const pow_neg1_c_plus_1 = ((c + 1) % 2 === 0 ? 1 : -1);
    const pow_neg1_r_c_1 = ((r + c + 1) % 2 === 0 ? 1 : -1);
    neighbors.push([r + pow_neg1_r, c + pow_neg1_c]);
    neighbors.push([r - 1, c - pow_neg1_r_c_1]);
    neighbors.push([r + 1, c + pow_neg1_r_c_1]);
    neighbors.push([r + pow_neg1_r_plus_1, c]);
    neighbors.push([r, c + pow_neg1_c_plus_1]);
    return neighbors.filter(([nr, nc]) => isValidCoord(nr, nc));
}

function checkGameOver() {
    if (placedTilesCount >= TOTAL_TILES) {
        console.log(`Board full (${placedTilesCount}/${TOTAL_TILES} tiles). Game Over!`);
        gameOver = true;
        calculateScore();
        updateScoreDisplay();
        updateTurnIndicator();
        showGameOverMessage();
        hoverPosition = null;
        canvas.style.cursor = 'default';
    } else {
        gameOver = false;
    }
}

function showGameOverMessage() {
    let message;
    const redScore = scores[RED];
    const blueScore = scores[BLUE];
    if (redScore > blueScore) {
        message = `Game Over! Red wins: ${redScore.toLocaleString()} to ${blueScore.toLocaleString()}`;
    } else if (blueScore > redScore) {
        message = `Game Over! Blue wins: ${blueScore.toLocaleString()} to ${redScore.toLocaleString()}`;
    } else {
        message = `Game Over! It's a tie: ${redScore.toLocaleString()}`;
    }
    console.log(message);
}

function updateTurnIndicator() {
    const turnIndicator = document.getElementById('turn-indicator');
    if (gameOver) {
        const redScore = scores[RED];
        const blueScore = scores[BLUE];
        let winnerText = "It's a Tie!";
        let winnerClass = "";
        if (typeof redScore !== 'number' || typeof blueScore !== 'number' || isNaN(redScore) || isNaN(blueScore)) {
            winnerText = "Game Over (Score Error)";
        } else if (redScore > blueScore) {
            winnerText = "Red Wins!"; 
            winnerClass = "red-text";
        } else if (blueScore > redScore) {
            winnerText = "Blue Wins!"; 
            winnerClass = "blue-text";
        }
        turnIndicator.innerHTML = `Game Over - <span class="${winnerClass}">${winnerText}</span>`;
    } else {
        const player = currentTurn === RED ? 'Red' : 'Blue';
        const playerClass = currentTurn === RED ? 'red-text' : 'blue-text';
        turnIndicator.innerHTML = `Current Turn: <span class="${playerClass}">${player}</span>`;
    }
}

function updateScoreDisplay() {
    const scoreDisplay = document.getElementById('score-display');
    const redScoreStr = (typeof scores[RED] === 'number' && !isNaN(scores[RED])) ? scores[RED].toLocaleString() : 'Error';
    const blueScoreStr = (typeof scores[BLUE] === 'number' && !isNaN(scores[BLUE])) ? scores[BLUE].toLocaleString() : 'Error';
    scoreDisplay.innerHTML = `
        <span class="red-text">Red: ${redScoreStr}</span> |
        <span class="blue-text">Blue: ${blueScoreStr}</span>
    `;
}

function getCoordsFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    let clientX, clientY;
    if (event.touches && event.touches.length > 0) {
        clientX = event.touches[0].clientX;
        clientY = event.touches[0].clientY;
    } else {
        clientX = event.clientX;
        clientY = event.clientY;
    }
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    return { x, y };
}

function getClickableCorner(x, y) {
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    const r = Math.floor(y / logicalCellSize);
    const c = Math.floor(x / logicalCellSize);
    const clampedR = Math.max(0, Math.min(LOGICAL_GRID_SIZE - 1, r));
    const clampedC = Math.max(0, Math.min(LOGICAL_GRID_SIZE - 1, c));
    if (isPlayable(clampedR, clampedC)) {
        return { r: clampedR, c: clampedC };
    }
    return null; 
}

function handleClick(event) {
    if (gameOver) return;
    const { x, y } = getCoordsFromEvent(event);
    const clickResult = getClickableCorner(x, y);
    if (clickResult) {
        const { r, c } = clickResult;
        console.log(`Click -> Playable Corner (${r}, ${c})`);
        const moveSuccessful = addTile(r, c);

        if (moveSuccessful) {
            calculateScore();
            updateScoreDisplay();
            currentTurn = 1 - currentTurn;
            updateTurnIndicator();
            checkGameOver();
            drawBoard();
            if (hoverPosition && hoverPosition.r === r && hoverPosition.c === c) {
                hoverPosition = null;
                canvas.style.cursor = 'default';
            }
        }
    } else {
        console.log("Click is not on a playable corner.");
    }
}

function handleMouseMove(event) {
    if (gameOver || !showPreview) {
        if (hoverPosition !== null) {
            hoverPosition = null;
            canvas.style.cursor = 'default';
            drawBoard();
        }
        return;
    }
    const { x, y } = getCoordsFromEvent(event);
    const potentialHover = getClickableCorner(x, y);
    const currentHoverStr = JSON.stringify(hoverPosition);
    const potentialHoverStr = JSON.stringify(potentialHover);
    if (currentHoverStr !== potentialHoverStr) {
        hoverPosition = potentialHover;
        canvas.style.cursor = hoverPosition ? 'pointer' : 'default';
        drawBoard(); 
    }
}

function handleMouseOut() {
    if (hoverPosition !== null && !gameOver) {
        hoverPosition = null;
        canvas.style.cursor = 'default';
        drawBoard();
    }
}

function resetGame() {
    console.log("Resetting game...");
    initializeBoard();
    updateScoreDisplay();
    updateTurnIndicator();
    canvas.style.cursor = 'default';
    drawBoard();
}

function togglePreview() {
    showPreview = !showPreview;
    if (!showPreview && hoverPosition !== null) {
        hoverPosition = null;
        canvas.style.cursor = 'default';
        drawBoard();
    } else if (showPreview) {
        handleMouseMove({ clientX: -1, clientY: -1 });
    }
}

function toggleRules() {
    const rulesPanel = document.getElementById('rules-panel');
    rulesPanel.style.display = rulesPanel.style.display === 'block' ? 'none' : 'block';
}

document.addEventListener('DOMContentLoaded', () => {
    initializeBoard();
    setupCanvas();
    updateScoreDisplay();
    updateTurnIndicator();
    
    canvas = document.getElementById('game-board');
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseout', handleMouseOut);
    canvas.addEventListener('touchstart', (e) => {
        if (gameOver) return;
        e.preventDefault();
        handleMouseMove(e);
    }, { passive: false });
    canvas.addEventListener('touchend', (e) => {
        if (gameOver) return;
        const currentHover = hoverPosition;
        if (currentHover) {
            handleClick(e);
        }
    }, { passive: false });
    
    document.getElementById('new-game-btn').addEventListener('click', resetGame);
    document.getElementById('rules-btn').addEventListener('click', toggleRules);
    
    window.addEventListener('resize', () => {
        console.log("Window resized");
        setupCanvas();
    });
});