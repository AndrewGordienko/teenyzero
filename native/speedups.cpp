#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace teenyzero_native {
PyObject* py_move_from_uci(PyObject*, PyObject*);
PyObject* py_move_uci(PyObject*, PyObject*);
PyObject* py_board_new(PyObject*, PyObject*);
PyObject* py_board_clone(PyObject*, PyObject*);
PyObject* py_board_turn(PyObject*, PyObject*);
PyObject* py_board_fullmove_number(PyObject*, PyObject*);
PyObject* py_board_halfmove_clock(PyObject*, PyObject*);
PyObject* py_board_ep_square(PyObject*, PyObject*);
PyObject* py_board_castling_rights(PyObject*, PyObject*);
PyObject* py_board_piece_masks(PyObject*, PyObject*);
PyObject* py_board_legal_moves(PyObject*, PyObject*);
PyObject* py_board_push(PyObject*, PyObject*);
PyObject* py_board_pop(PyObject*, PyObject*);
PyObject* py_board_move_stack(PyObject*, PyObject*);
PyObject* py_board_is_game_over(PyObject*, PyObject*);
PyObject* py_board_is_checkmate(PyObject*, PyObject*);
PyObject* py_board_result(PyObject*, PyObject*);
PyObject* py_board_can_claim_threefold_repetition(PyObject*, PyObject*);
PyObject* py_board_can_claim_fifty_moves(PyObject*, PyObject*);
PyObject* py_board_is_repetition(PyObject*, PyObject*);
PyObject* py_board_zobrist_hash(PyObject*, PyObject*);
PyObject* py_board_fen(PyObject*, PyObject*);
}  // namespace teenyzero_native

namespace {

int mirror_square(int square) {
    return square ^ 56;
}

PyObject* py_move_signature(PyObject*, PyObject* args) {
    int from_square = 0;
    int to_square = 0;
    int promotion = 0;
    if (!PyArg_ParseTuple(args, "ii|i", &from_square, &to_square, &promotion)) {
        return nullptr;
    }

    long signature = static_cast<long>(from_square)
        | (static_cast<long>(to_square) << 6)
        | (static_cast<long>(promotion) << 12);
    return PyLong_FromLong(signature);
}

PyObject* py_move_to_idx(PyObject*, PyObject* args) {
    int from_square = 0;
    int to_square = 0;
    int promotion = 0;
    int is_white_turn = 1;
    if (!PyArg_ParseTuple(args, "iiii", &from_square, &to_square, &promotion, &is_white_turn)) {
        return nullptr;
    }

    if (!is_white_turn) {
        from_square = mirror_square(from_square);
        to_square = mirror_square(to_square);
    }

    const int from_rank = from_square / 8;
    const int from_file = from_square % 8;
    const int to_rank = to_square / 8;
    const int to_file = to_square % 8;
    const int dr = to_rank - from_rank;
    const int df = to_file - from_file;

    if (promotion != 0 && promotion != 5) {
        int piece_offset = 0;
        if (promotion == 2) {
            piece_offset = 0;
        } else if (promotion == 3) {
            piece_offset = 1;
        } else if (promotion == 4) {
            piece_offset = 2;
        } else {
            PyErr_SetString(PyExc_ValueError, "unsupported promotion piece");
            return nullptr;
        }

        const int direction = df + 1;
        const int plane_idx = 64 + piece_offset * 3 + direction;
        return PyLong_FromLong(from_square * 73 + plane_idx);
    }

    static const int knight_moves[8][2] = {
        {2, 1}, {1, 2}, {-1, 2}, {-2, 1},
        {-2, -1}, {-1, -2}, {1, -2}, {2, -1},
    };
    for (int i = 0; i < 8; ++i) {
        if (dr == knight_moves[i][0] && df == knight_moves[i][1]) {
            return PyLong_FromLong(from_square * 73 + 56 + i);
        }
    }

    int dir_idx = -1;
    if (dr > 0 && df == 0) {
        dir_idx = 0;
    } else if (dr < 0 && df == 0) {
        dir_idx = 1;
    } else if (dr == 0 && df > 0) {
        dir_idx = 2;
    } else if (dr == 0 && df < 0) {
        dir_idx = 3;
    } else if (dr > 0 && df > 0) {
        dir_idx = 4;
    } else if (dr > 0 && df < 0) {
        dir_idx = 5;
    } else if (dr < 0 && df > 0) {
        dir_idx = 6;
    } else if (dr < 0 && df < 0) {
        dir_idx = 7;
    }

    if (dir_idx < 0) {
        PyErr_SetString(PyExc_ValueError, "unsupported move direction");
        return nullptr;
    }

    int abs_dr = dr >= 0 ? dr : -dr;
    int abs_df = df >= 0 ? df : -df;
    const int dist = abs_dr > abs_df ? abs_dr : abs_df;
    const int plane_idx = dir_idx * 7 + (dist - 1);
    return PyLong_FromLong(from_square * 73 + plane_idx);
}

PyMethodDef kMethods[] = {
    {"move_signature", py_move_signature, METH_VARARGS, "Pack a move into a compact integer signature."},
    {"move_to_idx", py_move_to_idx, METH_VARARGS, "Convert a move into the AlphaZero policy index."},
    {"move_from_uci", teenyzero_native::py_move_from_uci, METH_VARARGS, "Convert a UCI move into the packed move format."},
    {"move_uci", teenyzero_native::py_move_uci, METH_VARARGS, "Convert a packed move into UCI."},
    {"board_new", teenyzero_native::py_board_new, METH_VARARGS, "Create a native chess board."},
    {"board_clone", teenyzero_native::py_board_clone, METH_VARARGS, "Clone a native chess board."},
    {"board_turn", teenyzero_native::py_board_turn, METH_VARARGS, "Return side to move."},
    {"board_fullmove_number", teenyzero_native::py_board_fullmove_number, METH_VARARGS, "Return fullmove number."},
    {"board_halfmove_clock", teenyzero_native::py_board_halfmove_clock, METH_VARARGS, "Return halfmove clock."},
    {"board_ep_square", teenyzero_native::py_board_ep_square, METH_VARARGS, "Return en-passant square."},
    {"board_castling_rights", teenyzero_native::py_board_castling_rights, METH_VARARGS, "Return castling rights bitmask."},
    {"board_piece_masks", teenyzero_native::py_board_piece_masks, METH_VARARGS, "Return aggregate piece masks."},
    {"board_legal_moves", teenyzero_native::py_board_legal_moves, METH_VARARGS, "Return packed legal moves."},
    {"board_push", teenyzero_native::py_board_push, METH_VARARGS, "Push a packed move."},
    {"board_pop", teenyzero_native::py_board_pop, METH_VARARGS, "Pop the last move."},
    {"board_move_stack", teenyzero_native::py_board_move_stack, METH_VARARGS, "Return packed move stack."},
    {"board_is_game_over", teenyzero_native::py_board_is_game_over, METH_VARARGS, "Return whether the game is over."},
    {"board_is_checkmate", teenyzero_native::py_board_is_checkmate, METH_VARARGS, "Return whether the side to move is checkmated."},
    {"board_result", teenyzero_native::py_board_result, METH_VARARGS, "Return game result string."},
    {"board_can_claim_threefold_repetition", teenyzero_native::py_board_can_claim_threefold_repetition, METH_VARARGS, "Return whether threefold repetition can be claimed."},
    {"board_can_claim_fifty_moves", teenyzero_native::py_board_can_claim_fifty_moves, METH_VARARGS, "Return whether the fifty-move rule can be claimed."},
    {"board_is_repetition", teenyzero_native::py_board_is_repetition, METH_VARARGS, "Return repetition count test."},
    {"board_zobrist_hash", teenyzero_native::py_board_zobrist_hash, METH_VARARGS, "Return Zobrist hash."},
    {"board_fen", teenyzero_native::py_board_fen, METH_VARARGS, "Return FEN string."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_speedups",
    "Optional native helpers for TeenyZero hot paths.",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit__speedups(void) {
    return PyModule_Create(&kModule);
}
