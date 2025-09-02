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
let aiEnabled = false;
let aiSide = BLUE; 
let agentUrl = "https://tessellate-app-ytnku.ondigitalocean.app/";
let aiThinking = false;
let aiLoaderInterval = null;
let aiLoaderPhase = 0;
let aiStreamBuffer = '';
let aiFlushTimer = null;
let aiAutoScroll = true;

let canvas, ctx;
const colors = {
    [RED]: { normal: '#e94560', preview: '#ff99ad', glow: '#ffb3c1' },
    [BLUE]: { normal: '#3f72af', preview: '#9dc1ff', glow: '#c7dbff' },
    background: '#2b3a48',
    gridLines: '#c9d2dc',
    outOfPlay: '#22303a',
};

// Islands overlay state
let showIslands = true;
let islandsDirty = true;
let islandsData = null; // { islands, map, redSizes, blueSizes }
let hoveredIslandId = null;

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
    islandsDirty = true;
    islandsData = null;
    hoveredIslandId = null;
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
    islandsDirty = true; // pixel positions changed
    drawBoard();
    
    maybeMakeAIMove();
}

function drawBoard() {
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const tileState = board[r][c];
            if (tileState === RED || tileState === BLUE) {
                drawTile(r, c, tileState, false);
            } else if (tileState === OUT_OF_PLAY) {
                // Only dim out-of-play corners in EMPTY squares to avoid overlapping placed tiles
                const sr = Math.floor(r / 2), sc = Math.floor(c / 2);
                const corners = [
                    [2*sr, 2*sc], [2*sr, 2*sc+1], [2*sr+1, 2*sc], [2*sr+1, 2*sc+1]
                ];
                let squareHasTile = false;
                for (const [rr, cc] of corners) {
                    const v = board[rr][cc];
                    if (v === RED || v === BLUE) { squareHasTile = true; break; }
                }
                if (!squareHasTile) {
                    const pts = trianglePointsPx(r, c);
                    ctx.fillStyle = colors.outOfPlay;
                    ctx.beginPath();
                    ctx.moveTo(pts[0].x, pts[0].y);
                    ctx.lineTo(pts[1].x, pts[1].y);
                    ctx.lineTo(pts[2].x, pts[2].y);
                    ctx.closePath();
                    ctx.fill();
                }
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
    
    // Draw islands overlay after grid
    if (showIslands) {
        if (islandsDirty || !islandsData) islandsData = computeIslandsAndMap();
        if (islandsData) drawIslandsOverlay(islandsData);
    }

    if (showPreview && hoverPosition && !gameOver && isPlayable(hoverPosition.r, hoverPosition.c)) {
        drawTile(hoverPosition.r, hoverPosition.c, currentTurn, true);
        drawHoverHighlight(hoverPosition.r, hoverPosition.c, currentTurn);
    }
}

// ----- Agent View/Prompt Rendering -----
function rcToVisual(r, c) {
    const sr = Math.floor(r / 2);
    const sc = Math.floor(c / 2);
    const cr = r % 2;
    const cc = c % 2;
    const VIS_ROWS_LOCAL = ['A','B','C','D','E'];
    const corner = (cr === 0 && cc === 0) ? 'UL' : (cr === 0 && cc === 1) ? 'UR' : (cr === 1 && cc === 0) ? 'LL' : 'LR';
    return `${VIS_ROWS_LOCAL[sr]}${sc}-${corner}`;
}

function renderBoardAscii() {
    const RED_TRI = { 'UL': '◤', 'UR': '◥', 'LL': '◣', 'LR': '◢' };
    const BLUE_TRI = { 'UL': '◸', 'UR': '◹', 'LL': '◺', 'LR': '◿' };

    function cornerChar(r, c) {
        const val = board[r][c];
        const corner = (r % 2 === 0 && c % 2 === 0) ? 'UL' : (r % 2 === 0 && c % 2 === 1) ? 'UR' : (r % 2 === 1 && c % 2 === 0) ? 'LL' : 'LR';
        if (val === RED) return RED_TRI[corner];
        if (val === BLUE) return BLUE_TRI[corner];
        return '·';
    }

    const VIS_ROWS_LOCAL = ['A','B','C','D','E'];
    const lines = [];
    lines.push('    ' + [0,1,2,3,4].join('  '));
    for (let sr = 0; sr < 5; sr++) {
        const topCells = [];
        const botCells = [];
        for (let sc = 0; sc < 5; sc++) {
            const ul = cornerChar(2*sr, 2*sc);
            const ur = cornerChar(2*sr, 2*sc+1);
            const ll = cornerChar(2*sr+1, 2*sc);
            const lr = cornerChar(2*sr+1, 2*sc+1);
            topCells.push(`${ul}${ur}`);
            botCells.push(`${ll}${lr}`);
        }
        lines.push(`${VIS_ROWS_LOCAL[sr].padStart(2, ' ')}  ` + topCells.join(' '));
        lines.push('    ' + botCells.join(' '));
        if (sr < 4) lines.push('');
    }
    return lines.join('\n');
}

function buildAgentPromptFromCurrentState() {
    const boardStr = renderBoardAscii();
    const turnStr = (currentTurn === RED) ? 'RED' : 'BLUE';
    const valid = getValidActionsFlat();
    const items = [];
    for (let i = 0; i < valid.length && i < 50; i++) {
        const a = valid[i];
        const r = Math.floor(a / 10);
        const c = a % 10;
        items.push(rcToVisual(r, c));
    }
    if (valid.length > 50) items.push('...');
    const movesVisual = items.join(', ');
    // Island/score summary for clarity
    const islands = computeIslandSummary();
    const redIslands = islands.red.sizes.join('×') || '–';
    const blueIslands = islands.blue.sizes.join('×') || '–';
    const islandSummary = `Islands — Red: [${redIslands}] => ${islands.red.product}; Blue: [${blueIslands}] => ${islands.blue.product}`;

    const instructions = (
        'You are playing Tessellate. Place one triangular tile per turn.\n' +
        'Respond in two labeled parts so we can show your thinking without private chain-of-thought.\n' +
        '- Move: a single coordinate (prefer [A0] or visual A0-UR).\n' +
        '- Reasoning: a concise high-level rationale (1-2 lines).\n' +
        'Valid row letters: A-J (logical grid), or A-E with UL/UR/LL/LR for the visual squares. Choose only empty (not blocked) positions.\n'
    );
    const msg = (
        boardStr + '\n' +
        islandSummary + '\n' +
        `Turn: ${turnStr}.\n` +
        `Valid moves (visual, up to 50): ${movesVisual}\n\n` +
        instructions
    );
    return msg;
}

function updateAiViewPrompt() {
    const viewEl = document.getElementById('ai-view');
    if (!viewEl) return;
    viewEl.textContent = buildAgentPromptFromCurrentState();
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
    // Soft, pronounced border using a darker shade of the tile color
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.miterLimit = 2;
    ctx.lineWidth = Math.max(1.2, visualCellSize * 0.05);
    const darkStroke = (color === RED) ? 'rgba(120,30,45,0.85)' : 'rgba(45,72,120,0.85)';
    ctx.shadowColor = darkStroke;
    ctx.shadowBlur = Math.max(0.5, visualCellSize * 0.12);
    ctx.strokeStyle = darkStroke;
    ctx.stroke();
    ctx.restore();
}

function trianglePointsPx(r, c) {
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
    const isRowEven = r % 2 === 0;
    const isColEven = c % 2 === 0;
    if (isRowEven && isColEven) return [tl, tr, bl];
    if (isRowEven && !isColEven) return [tr, br, tl];
    if (!isRowEven && isColEven) return [bl, tl, br];
    return [br, bl, tr];
}

function triangleCentroidPx(r, c) {
    const pts = trianglePointsPx(r, c);
    const x = (pts[0].x + pts[1].x + pts[2].x) / 3;
    const y = (pts[0].y + pts[1].y + pts[2].y) / 3;
    return { x, y };
}

function rightAngleLabelPos(r, c) {
    // Position near the square corner that is the triangle's right angle,
    // inset slightly along both legs to nestle inside.
    const visualY = Math.floor(r / 2);
    const visualX = Math.floor(c / 2);
    const x0 = visualX * visualCellSize;
    const y0 = visualY * visualCellSize;
    const x1 = (visualX + 1) * visualCellSize;
    const y1 = (visualY + 1) * visualCellSize;
    const isRowEven = r % 2 === 0;
    const isColEven = c % 2 === 0;
    const inset = Math.max(visualCellSize * 0.225, 4); // nudge inside
    let x, y;
    if (isRowEven && isColEven) { // UL
        x = x0 + inset; y = y0 + inset;
    } else if (isRowEven && !isColEven) { // UR
        x = x1 - inset; y = y0 + inset;
    } else if (!isRowEven && isColEven) { // LL
        x = x0 + inset; y = y1 - inset;
    } else { // LR
        x = x1 - inset; y = y1 - inset;
    }
    return { x, y };
}

function hexToRgba(hex, alpha) {
    const h = hex.replace('#','');
    const bigint = parseInt(h, 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return `rgba(${r},${g},${b},${alpha})`;
}

function shadeHex(hex, factor) {
    const h = hex.replace('#','');
    const bigint = parseInt(h, 16);
    let r = (bigint >> 16) & 255;
    let g = (bigint >> 8) & 255;
    let b = bigint & 255;
    r = Math.max(0, Math.min(255, Math.round(r * factor)));
    g = Math.max(0, Math.min(255, Math.round(g * factor)));
    b = Math.max(0, Math.min(255, Math.round(b * factor)));
    return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

function computeIslandsAndMap() {
    const visited = Array(LOGICAL_GRID_SIZE).fill(null).map(() => Array(LOGICAL_GRID_SIZE).fill(false));
    const idMap = Array(LOGICAL_GRID_SIZE).fill(null).map(() => Array(LOGICAL_GRID_SIZE).fill(-1));
    const islands = [];
    const redSizes = [];
    const blueSizes = [];
    let nextId = 0;

    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const color = board[r][c];
            if ((color === RED || color === BLUE) && !visited[r][c]) {
                const cells = [];
                const centroids = [];
                let size = 0;
                const stack = [[r, c]];
                while (stack.length) {
                    const [rr, cc] = stack.pop();
                    if (rr < 0 || rr >= LOGICAL_GRID_SIZE || cc < 0 || cc >= LOGICAL_GRID_SIZE) continue;
                    if (visited[rr][cc] || board[rr][cc] !== color) continue;
                    visited[rr][cc] = true;
                    idMap[rr][cc] = nextId;
                    size++;
                    cells.push([rr, cc]);
                    centroids.push(triangleCentroidPx(rr, cc));
                    const neigh = getNeighbors(rr, cc);
                    for (const [nr, nc] of neigh) {
                        if (!visited[nr][nc] && board[nr][nc] === color) stack.push([nr, nc]);
                    }
                }
                const avgX = centroids.reduce((p,a)=>p+a.x,0) / centroids.length;
                const avgY = centroids.reduce((p,a)=>p+a.y,0) / centroids.length;
                // Choose the triangle whose centroid is closest to the island average
                let bestIdx = 0, bestDist = Infinity;
                for (let i = 0; i < centroids.length; i++) {
                    const dx = centroids[i].x - avgX;
                    const dy = centroids[i].y - avgY;
                    const d2 = dx*dx + dy*dy;
                    if (d2 < bestDist) { bestDist = d2; bestIdx = i; }
                }
                const anchorCell = cells[bestIdx];
                const labelPos = rightAngleLabelPos(anchorCell[0], anchorCell[1]);
                islands.push({ id: nextId, color, size, cells, labelPos, centroids });
                if (color === RED) redSizes.push(size); else blueSizes.push(size);
                nextId++;
            }
        }
    }
    const sortDesc = arr => arr.sort((a,b)=>b-a);
    sortDesc(redSizes); sortDesc(blueSizes);
    islandsDirty = false;
    return { islands, map: idMap, redSizes, blueSizes };
}

function drawIslandsOverlay(data) {
    // semi-transparent fills and size labels; emphasize current player's islands and hovered island
    for (const isl of data.islands) {
        const alpha = (isl.id === hoveredIslandId) ? 0.50 : (isl.color === currentTurn ? 0.22 : 0.10);
        const base = isl.color === RED ? colors[RED].normal : colors[BLUE].normal;
        const dark = shadeHex(base, 0.6);
        // Build a single path for the whole island to avoid internal edges
        const path = new Path2D();
        for (const [r,c] of isl.cells) {
            const pts = trianglePointsPx(r,c);
            path.moveTo(pts[0].x, pts[0].y);
            path.lineTo(pts[1].x, pts[1].y);
            path.lineTo(pts[2].x, pts[2].y);
            path.closePath();
        }
        // Tint fill
        ctx.fillStyle = hexToRgba(base, alpha);
        ctx.fill(path);
        // Darker soft outline around island perimeter
        ctx.save();
        ctx.shadowColor = hexToRgba(dark, 0.9);
        ctx.shadowBlur = Math.max(6, visualCellSize * 0.18);
        ctx.fillStyle = 'rgba(0,0,0,0)';
        ctx.fill(path);
        ctx.restore();
        // Crisp perimeter stroke
        ctx.save();
        ctx.lineJoin = 'round';
        ctx.lineWidth = Math.max(1.2, visualCellSize * 0.035);
        ctx.strokeStyle = hexToRgba(dark, 0.9);
        ctx.stroke(path);
        ctx.restore();
        // Stronger outline for hovered island
        if (isl.id === hoveredIslandId) {
            ctx.lineJoin = 'round';
            ctx.lineWidth = Math.max(2, visualCellSize * 0.08);
            ctx.strokeStyle = hexToRgba(base, 0.95);
            ctx.shadowColor = hexToRgba(base, 0.95);
            ctx.shadowBlur = Math.max(8, visualCellSize * 0.35);
            ctx.stroke(path);
            ctx.shadowBlur = 0;
        }
        // Label
        const label = String(isl.size);
        ctx.font = `${Math.round(visualCellSize*0.32)}px Inter, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(0,0,0,0.75)';
        ctx.strokeText(label, isl.labelPos.x, isl.labelPos.y);
        ctx.fillStyle = isl.color === RED ? '#ffd3da' : '#d8e6ff';
        ctx.fillText(label, isl.labelPos.x, isl.labelPos.y);
    }
}

function drawHoverHighlight(r, c, color) {
    const pts = trianglePointsPx(r, c);
    const glow = color === RED ? colors[RED].glow : colors[BLUE].glow;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    ctx.lineTo(pts[1].x, pts[1].y);
    ctx.lineTo(pts[2].x, pts[2].y);
    ctx.closePath();
    // Outer glow
    ctx.shadowColor = glow;
    ctx.shadowBlur = Math.max(12, visualCellSize * 0.35);
    ctx.fillStyle = 'rgba(255,255,255,0.06)';
    ctx.fill();
    // Inner stroke highlight
    ctx.shadowBlur = 0;
    ctx.lineWidth = Math.max(2, visualCellSize * 0.06);
    ctx.strokeStyle = glow;
    ctx.stroke();
    ctx.restore();
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
    islandsDirty = true;
    
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
    islandsDirty = true;
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
    const factorsEl = document.getElementById('score-factors');
    if (factorsEl) {
        if (showIslands) {
            if (islandsDirty || !islandsData) islandsData = computeIslandsAndMap();
            const redList = (islandsData && islandsData.redSizes.length) ? islandsData.redSizes.join('×') : '–';
            const blueList = (islandsData && islandsData.blueSizes.length) ? islandsData.blueSizes.join('×') : '–';
            const redProd = (islandsData && islandsData.redSizes.length) ? islandsData.redSizes.reduce((p,x)=>p*x,1) : 1;
            const blueProd = (islandsData && islandsData.blueSizes.length) ? islandsData.blueSizes.reduce((p,x)=>p*x,1) : 1;
            factorsEl.innerHTML = `Red: [${redList}] = <span class="red-text">${redProd}</span> | Blue: [${blueList}] = <span class="blue-text">${blueProd}</span>`;
            factorsEl.style.display = '';
        } else {
            factorsEl.style.display = 'none';
            factorsEl.textContent = '';
        }
    }
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

function getCornerAtXY(x, y) {
    if (!canvas) return null;
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    const r = Math.floor(y / logicalCellSize);
    const c = Math.floor(x / logicalCellSize);
    const clampedR = Math.max(0, Math.min(LOGICAL_GRID_SIZE - 1, r));
    const clampedC = Math.max(0, Math.min(LOGICAL_GRID_SIZE - 1, c));
    return { r: clampedR, c: clampedC };
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
    if (aiEnabled && currentTurn === aiSide) return; 
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
            updateAiViewPrompt();
            if (hoverPosition && hoverPosition.r === r && hoverPosition.c === c) {
                hoverPosition = null;
                canvas.style.cursor = 'default';
            }
            maybeMakeAIMove();
        }
    } else {
        console.log("Click is not on a playable corner.");
    }
}

function handleMouseMove(event) {
    // Always track hovered island independent of preview
    const { x, y } = getCoordsFromEvent(event);
    if (showIslands && (islandsDirty || !islandsData)) islandsData = computeIslandsAndMap();
    if (showIslands && islandsData) {
        const rcAny = getCornerAtXY(x, y);
        if (rcAny) {
            const id = islandsData.map[rcAny.r][rcAny.c];
            const newHover = (typeof id === 'number' && id >= 0) ? id : null;
            if (newHover !== hoveredIslandId) { hoveredIslandId = newHover; drawBoard(); }
        }
    }

    if (gameOver || !showPreview) {
        if (hoverPosition !== null) {
            hoverPosition = null;
            canvas.style.cursor = 'default';
            drawBoard();
        }
        return;
    }
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
    const expEl = document.getElementById('ai-explanation');
    const loaderEl = document.getElementById('ai-loader');
    const streamEl = document.getElementById('ai-stream');
    if (expEl) { expEl.style.display = 'none'; }
    if (loaderEl) loaderEl.textContent = '';
    if (streamEl) streamEl.textContent = '';
    updateAiViewPrompt();
    
    maybeMakeAIMove();
}

function startAiLoader() {
    const expEl = document.getElementById('ai-explanation');
    const loaderEl = document.getElementById('ai-loader');
    if (!expEl || !loaderEl) return;
    expEl.dataset.streaming = '1';
    const colorLabel = (aiSide === RED) ? 'Red' : 'Blue';
    aiLoaderPhase = 0;
    const frames = ['.', '..', '...'];
    expEl.style.display = 'block';
    // Only update loader area; keep streamed text separate
    const streamEl = document.getElementById('ai-stream');
    if (streamEl && streamEl.textContent && !streamEl.textContent.endsWith('\n')) {
        streamEl.textContent += '\n';
    }
    loaderEl.textContent = `${colorLabel} AI is thinking`;
    if (aiLoaderInterval) clearInterval(aiLoaderInterval);
    aiLoaderInterval = setInterval(() => {
        aiLoaderPhase = (aiLoaderPhase + 1) % frames.length;
        loaderEl.textContent = `${colorLabel} AI is thinking${frames[aiLoaderPhase]}`;
        if (aiAutoScroll) expEl.scrollTop = expEl.scrollHeight;
    }, 400);
}

function stopAiLoader() {
    if (aiLoaderInterval) {
        clearInterval(aiLoaderInterval);
        aiLoaderInterval = null;
    }
    const loaderEl = document.getElementById('ai-loader');
    if (loaderEl) loaderEl.textContent = '';
}

function scheduleAiFlush() {
    if (aiFlushTimer) return;
    aiFlushTimer = setTimeout(() => {
        const expEl = document.getElementById('ai-explanation');
        const streamEl = document.getElementById('ai-stream');
        if (expEl && streamEl && aiStreamBuffer) {
            streamEl.textContent += aiStreamBuffer;
            aiStreamBuffer = '';
            if (aiAutoScroll) expEl.scrollTop = expEl.scrollHeight;
        }
        aiFlushTimer = null;
    }, 60);
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
    const isHidden = rulesPanel.getAttribute('aria-hidden') !== 'false';
    const nextHidden = !isHidden ? 'true' : 'false';
    rulesPanel.setAttribute('aria-hidden', nextHidden);
    rulesPanel.style.display = nextHidden === 'false' ? 'block' : 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    initializeBoard();
    setupCanvas();
    updateScoreDisplay();
    updateTurnIndicator();
    updateAiViewPrompt();
    
    canvas = document.getElementById('game-board');
    canvas.addEventListener('click', handleClick);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseout', handleMouseOut);
    canvas.addEventListener('touchstart', (e) => {
        if (gameOver) return;
    if (aiEnabled && currentTurn === aiSide) return; 
        e.preventDefault();
        handleMouseMove(e);
    }, { passive: false });
    canvas.addEventListener('touchend', (e) => {
        if (gameOver) return;
    if (aiEnabled && currentTurn === aiSide) return; 
        const currentHover = hoverPosition;
        if (currentHover) {
            handleClick(e);
        }
    }, { passive: false });
    
    document.getElementById('new-game-btn').addEventListener('click', resetGame);
    document.getElementById('rules-btn').addEventListener('click', toggleRules);
    const aiEnabledEl = document.getElementById('ai-enabled');
    const aiSideEl = document.getElementById('ai-side');
    const agentUrlEl = document.getElementById('agent-url');
    const islandsToggleEl = document.getElementById('islands-toggle');
    function sanitizeAgentUrl(v) {
        if (!v) return '';
        let s = String(v).trim();
        if (!/^https?:\/\//i.test(s)) s = 'http://' + s;
        // remove any trailing slashes
        s = s.replace(/\/+$/, '');
        return s;
    }

    if (aiEnabledEl && aiSideEl && agentUrlEl) {
        aiEnabledEl.addEventListener('change', (e) => {
            aiEnabled = e.target.checked;
            if (aiEnabled) {
                aiSide = (aiSideEl.value === 'RED') ? RED : BLUE;
                agentUrl = sanitizeAgentUrl(agentUrlEl.value) || agentUrl;
                updateAiViewPrompt();
                maybeMakeAIMove();
            }
        });
        aiSideEl.addEventListener('change', (e) => {
            aiSide = (e.target.value === 'RED') ? RED : BLUE;
            updateAiViewPrompt();
            maybeMakeAIMove();
        });
        agentUrlEl.addEventListener('change', (e) => {
        const cleaned = sanitizeAgentUrl(e.target.value);
        if (cleaned) {
            agentUrl = cleaned;
            e.target.value = cleaned;
        }
        });
    }
    if (islandsToggleEl) {
        islandsToggleEl.addEventListener('change', (e) => {
            showIslands = !!e.target.checked;
            drawBoard();
            updateScoreDisplay();
        });
    }
    document.addEventListener('keydown', (e) => {
        if (!e.key) return;
        const k = e.key.toLowerCase();
        const target = e.target;
        const isTyping = target && ((target.tagName === 'INPUT' && !['checkbox','radio','button','submit','reset'].includes(target.type)) || target.tagName === 'TEXTAREA' || target.isContentEditable);
        if (isTyping) return; // do not hijack typing into fields
        if (k === 'i') {
            e.preventDefault();
            showIslands = !showIslands;
            if (islandsToggleEl) islandsToggleEl.checked = showIslands;
            drawBoard();
            updateScoreDisplay();
        } else if (k === 'a') {
            e.preventDefault();
            if (aiEnabledEl) {
                aiEnabledEl.checked = !aiEnabledEl.checked;
                aiEnabledEl.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    });
    
    window.addEventListener('resize', () => {
        console.log("Window resized");
        setupCanvas();
    });
    const expEl2 = document.getElementById('ai-explanation');
    if (expEl2) {
        expEl2.addEventListener('scroll', () => {
            const nearBottom = expEl2.scrollHeight - expEl2.clientHeight - expEl2.scrollTop < 8;
            aiAutoScroll = nearBottom;
        });
    }

    // Tabs wiring
    // Collapsible sections wiring
    const collapseThinking = document.getElementById('collapse-thinking');
    const collapseView = document.getElementById('collapse-view');
    const thinkingContent = document.getElementById('thinking-content');
    const viewContent = document.getElementById('view-content');
    if (collapseThinking && thinkingContent) {
        collapseThinking.addEventListener('click', () => {
            const expanded = collapseThinking.getAttribute('aria-expanded') === 'true';
            const next = !expanded;
            collapseThinking.setAttribute('aria-expanded', next ? 'true' : 'false');
            collapseThinking.textContent = next ? '▾' : '▸';
            thinkingContent.style.display = next ? '' : 'none';
        });
    }
    if (collapseView && viewContent) {
        collapseView.addEventListener('click', () => {
            const expanded = collapseView.getAttribute('aria-expanded') === 'true';
            const next = !expanded;
            collapseView.setAttribute('aria-expanded', next ? 'true' : 'false');
            collapseView.textContent = next ? '▾' : '▸';
            viewContent.style.display = next ? '' : 'none';
        });
    }
});

// Build model-compatible state vector (length 104)
function buildModelState() {
    const obs = new Array(104).fill(0);
    let idx = 0;
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const v = board[r][c];
            let mapped = 0;
            if (v === RED) mapped = 1;
            else if (v === BLUE) mapped = 2;
            else if (v === EMPTY) mapped = 0;
            else if (v === OUT_OF_PLAY) mapped = 3;
            obs[idx++] = mapped;
        }
    }
    obs[100] = (currentTurn === RED) ? 1 : 2;
    obs[101] = scores[RED] || 1;
    obs[102] = scores[BLUE] || 1;
    obs[103] = placedTilesCount;
    return obs;
}

function getValidActionsFlat() {
    const acts = [];
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            if (board[r][c] === EMPTY) acts.push(r * 10 + c);
        }
    }
    return acts;
}

async function maybeMakeAIMove() {
    if (!aiEnabled || gameOver || currentTurn !== aiSide || aiThinking) return;
    aiThinking = true;
    try {
        const state = buildModelState();
        const valid_actions = getValidActionsFlat();
        if (valid_actions.length === 0) { aiThinking = false; return; }
        let action = -1;
        let explanation = '';
        const useBrowser = !agentUrl || agentUrl.trim() === '' || agentUrl.startsWith('browser');
        startAiLoader();
        if (useBrowser && window.PQN) {
            const modelUrl = 'model/pqn_model_batch50.json';
            action = await window.PQN.selectAction(modelUrl, state, valid_actions);
        } else {
            // Try streaming endpoint first
            try {
                const res = await fetch(`${agentUrl}/move_stream`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'text/event-stream'
                    },
                    body: JSON.stringify({ state, valid_actions })
                });
                if (!res.ok || !res.body) throw new Error(`Streaming not available: ${res.status}`);
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                const expEl = document.getElementById('ai-explanation');
                if (expEl) { expEl.style.display = 'block'; }
                let sseEvent = null;
                let sseData = '';
                const processEvent = (eventName, dataStr) => {
                    const payload = dataStr.trim();
                    if (!payload) return;
                    if (payload === '[DONE]') return;
                    // Try typed envelope first
                    try {
                        const obj = JSON.parse(payload);
                        if (obj && typeof obj === 'object' && 'type' in obj) {
                            if ((obj.type === 'content' || obj.type === 'reasoning') && typeof obj.delta === 'string') {
                                aiStreamBuffer += obj.delta;
                                scheduleAiFlush();
                                return;
                            }
                            if (obj.type === 'final') {
                                action = obj.action;
                                explanation = obj.full || '';
                                return;
                            }
                            if (obj.type === 'info' && obj.message) {
                                aiStreamBuffer += `\n\n[system] ${obj.message}`;
                                scheduleAiFlush();
                                return;
                            }
                            if (obj.type === 'error') {
                                throw new Error(obj.message || 'stream error');
                            }
                        }
                    } catch {}
                    // Fallback: OpenAI/OpenRouter raw delta format
                    try {
                        const obj = JSON.parse(payload);
                        const choice = obj.choices && obj.choices[0];
                        const delta = choice && choice.delta ? choice.delta : {};
                        // Reasoning token support
                        let reasonTxt = null;
                        if (delta.reasoning) {
                            if (typeof delta.reasoning === 'string') reasonTxt = delta.reasoning;
                            else if (typeof delta.reasoning === 'object') {
                                reasonTxt = delta.reasoning.text || delta.reasoning.content || null;
                            }
                        }
                        if (reasonTxt) {
                            aiStreamBuffer += reasonTxt;
                            scheduleAiFlush();
                        }
                        const contentTxt = (typeof delta.content === 'string') ? delta.content : null;
                        if (contentTxt) {
                            aiStreamBuffer += contentTxt;
                            scheduleAiFlush();
                        }
                        return;
                    } catch {}
                    // last resort: append raw
                    aiStreamBuffer += payload;
                    scheduleAiFlush();
                };

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n')) !== -1) {
                        const rawLine = buffer.slice(0, idx);
                        buffer = buffer.slice(idx + 1);
                        if (rawLine === '') {
                            if (sseData) {
                                processEvent(sseEvent, sseData);
                                sseEvent = null;
                                sseData = '';
                            }
                            continue;
                        }
                        if (rawLine.startsWith('event:')) {
                            sseEvent = rawLine.slice(6).trim();
                            continue;
                        }
                        if (rawLine.startsWith('data:')) {
                            sseData += rawLine.slice(5).trimStart() + '\n';
                            continue;
                        }
                        // comments/other fields ignored
                    }
                    if (action !== -1 && typeof action === 'number') break;
                }
                try { reader.cancel(); } catch {}
                if (typeof action !== 'number' || !valid_actions.includes(action)) {
                    // If stream doesn't supply final, request non-stream final action
                    const res2 = await fetch(`${agentUrl}/move`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ state, valid_actions })
                    });
                    if (!res2.ok) {
                        const expEl2 = document.getElementById('ai-explanation');
                        const streamEl2 = document.getElementById('ai-stream');
                        let errText = `${res2.status} ${res2.statusText}`;
                        try { const t = await res2.text(); if (t) errText = `${errText} - ${t}`; } catch {}
                        if (expEl2 && streamEl2) {
                            expEl2.style.display = 'block';
                            streamEl2.textContent += `\n\nAI error: ${errText}`;
                        }
                        return;
                    }
                    const data = await res2.json();
                    action = (data && typeof data.action === 'number') ? data.action : -1;
                    if (data && typeof data.explanation === 'string') explanation = data.explanation;
                }
            } catch (streamErr) {
                // Fallback to non-streaming JSON endpoint
                const res2 = await fetch(`${agentUrl}/move`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state, valid_actions })
                });
                if (!res2.ok) {
                    const expEl = document.getElementById('ai-explanation');
                    const streamEl = document.getElementById('ai-stream');
                    let errText = `${res2.status} ${res2.statusText}`;
                    try { const t = await res2.text(); if (t) errText = `${errText} - ${t}`; } catch {}
                    if (expEl && streamEl) {
                        expEl.style.display = 'block';
                        streamEl.textContent = `AI error: ${errText}`;
                    }
                    return;
                }
                const data = await res2.json();
                action = (data && typeof data.action === 'number') ? data.action : -1;
                if (data && typeof data.explanation === 'string') explanation = data.explanation;
            }
        }
        if (!valid_actions.includes(action)) {
            console.warn('Agent returned invalid action.');
            const expEl = document.getElementById('ai-explanation');
            const streamEl = document.getElementById('ai-stream');
            if (expEl && streamEl) {
                expEl.style.display = 'block';
                streamEl.textContent = 'AI error: returned invalid action.';
            }
            return; // do not fallback randomly
        }
        const r = Math.floor(action / 10);
        const c = action % 10;
        if (isPlayable(r, c)) {
            addTile(r, c);
            calculateScore();
            updateScoreDisplay();
            currentTurn = 1 - currentTurn;
            updateTurnIndicator();
            checkGameOver();
            drawBoard();
            updateAiViewPrompt();
            // Do not overwrite streamed content; append a concise end-of-turn marker instead
            const expEl = document.getElementById('ai-explanation');
            const streamEl = document.getElementById('ai-stream');
            if (expEl && streamEl) {
                expEl.style.display = 'block';
                // const colorLabel = (aiSide === RED) ? 'Red' : 'Blue';
                // streamEl.textContent += `\n\n— ${colorLabel} move applied.`;
                if (aiAutoScroll) expEl.scrollTop = expEl.scrollHeight;
            }
        }
    } catch (e) {
        console.error('AI move error:', e);
        const expEl = document.getElementById('ai-explanation');
        const streamEl = document.getElementById('ai-stream');
        if (expEl && streamEl) {
            expEl.style.display = 'block';
            streamEl.textContent = `AI error: ${e && e.message ? e.message : 'Failed to fetch'}`;
        }
    } finally {
        stopAiLoader();
        aiThinking = false;
    }
}

// ----- Island summary for Agent View -----
function computeIslandSummary() {
    // Collect island sizes for each color using existing DFS utilities
    const visited = Array(LOGICAL_GRID_SIZE).fill(null).map(() => Array(LOGICAL_GRID_SIZE).fill(false));
    const sizes = { [RED]: [], [BLUE]: [] };
    for (let r = 0; r < LOGICAL_GRID_SIZE; r++) {
        for (let c = 0; c < LOGICAL_GRID_SIZE; c++) {
            const color = board[r][c];
            if ((color === RED || color === BLUE) && !visited[r][c]) {
                const size = dfs(r, c, color, visited);
                if (size > 0) sizes[color].push(size);
            }
        }
    }
    const sortDesc = arr => arr.slice().sort((a,b) => b-a);
    const redSizes = sortDesc(sizes[RED]);
    const blueSizes = sortDesc(sizes[BLUE]);
    const prod = arr => arr.reduce((p,x) => p*x, 1) || 1;
    return {
        red: { sizes: redSizes, product: prod(redSizes) },
        blue: { sizes: blueSizes, product: prod(blueSizes) },
    };
}
