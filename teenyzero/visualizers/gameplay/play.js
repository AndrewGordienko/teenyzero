let board = null;
let game = new Chess();
let playerSide = "w";

function syncGame(fen) {
    game.load(fen);
    board.position(fen, false);
}

function onDragStart(source, piece) {
    const pieceSide = piece[0].toLowerCase();
    if (game.game_over()) return false;
    if (pieceSide !== playerSide) return false;
    if ((game.turn() === "w" && playerSide !== "w") || (game.turn() === "b" && playerSide !== "b")) return false;
    return true;
}

function onDrop(source, target) {
    if (source === target) return "snapback";

    const move = game.move({
        from: source,
        to: target,
        promotion: "q",
    });

    if (move === null) {
        return "snapback";
    }

    $("#status").text("Computing...");

    $.ajax({
        url: "/move",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ uci: move.from + move.to + (move.promotion || "") }),
        success(data) {
            if (data.fen) {
                syncGame(data.fen);
            }
            if (data.win_prob !== undefined) {
                $("#win-prob").text(`${Number(data.win_prob).toFixed(1)}%`);
            }
            $("#status").text(game.turn() === playerSide ? "Your Move" : "Engine Move");
        },
        error(err) {
            console.error("Move Error:", err);
            game.undo();
            board.position(game.fen(), false);
            $("#status").text("Illegal Move");
        },
    });
}

function resetGame() {
    playerSide = $("#user-side").val();
    $("#status").text("Resetting...");

    $.ajax({
        url: "/reset",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ side: playerSide }),
        success(data) {
            board.orientation(playerSide === "w" ? "white" : "black");
            syncGame(data.fen);
            $("#win-prob").text(`${Number(data.win_prob || 50.0).toFixed(1)}%`);
            $("#status").text(game.turn() === playerSide ? "Your Move" : "Engine Move");
        },
        error(err) {
            console.error("Reset Error:", err);
            $("#status").text("Reset Failed");
        },
    });
}

board = Chessboard("board", {
    draggable: true,
    position: "start",
    onDragStart,
    onDrop,
    pieceTheme(piece) {
        return `/static/assets/pieces/${String(piece).toLowerCase()}.png`;
    },
});

document.getElementById("user-side").addEventListener("change", resetGame);
document.getElementById("reset-button").addEventListener("click", resetGame);

resetGame();
