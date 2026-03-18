from __future__ import annotations

from dataclasses import dataclass

from teenyzero.native import speedups


WHITE = True
BLACK = False

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

CASTLE_WHITE_KINGSIDE = 1
CASTLE_WHITE_QUEENSIDE = 2
CASTLE_BLACK_KINGSIDE = 4
CASTLE_BLACK_QUEENSIDE = 8


def _pack_move_from_parts(from_square, to_square, promotion=0):
    return int(from_square) | (int(to_square) << 6) | (int(promotion or 0) << 12)


def _coerce_packed_move(move):
    if isinstance(move, NativeMove):
        return move.packed
    if isinstance(move, int):
        return int(move)
    promotion = int(getattr(move, "promotion", 0) or 0)
    return _pack_move_from_parts(move.from_square, move.to_square, promotion)


@dataclass(frozen=True)
class NativeMove:
    packed: int

    @property
    def from_square(self):
        return self.packed & 63

    @property
    def to_square(self):
        return (self.packed >> 6) & 63

    @property
    def promotion(self):
        promotion = (self.packed >> 12) & 7
        return promotion or None

    def uci(self):
        return speedups.move_uci(int(self.packed))

    @classmethod
    def from_uci(cls, uci):
        return cls(int(speedups.move_from_uci(uci)))

    def __int__(self):
        return int(self.packed)

    def __repr__(self):
        return f"NativeMove({self.uci()!r})"


class NativeLegalMoveView:
    __slots__ = ("_board",)

    def __init__(self, board):
        self._board = board

    def __iter__(self):
        return iter(self._board._legal_moves_tuple())

    def __len__(self):
        return len(self._board._legal_moves_tuple())

    def __contains__(self, move):
        return NativeMove(_coerce_packed_move(move)) in self._board._legal_moves_tuple()

    def count(self):
        return len(self)


class NativeBoard:
    __slots__ = (
        "_capsule",
        "_version",
        "_legal_moves_cache",
        "_legal_moves_version",
        "_piece_masks_cache",
        "_piece_masks_version",
        "legal_moves",
    )

    is_native = True

    def __init__(self, fen=None, _capsule=None):
        self._capsule = _capsule if _capsule is not None else speedups.board_new(fen)
        self._version = 0
        self._legal_moves_cache = None
        self._legal_moves_version = -1
        self._piece_masks_cache = None
        self._piece_masks_version = -1
        self.legal_moves = NativeLegalMoveView(self)

    def _invalidate(self):
        self._version += 1
        self._legal_moves_cache = None
        self._piece_masks_cache = None

    def _piece_masks(self):
        if self._piece_masks_cache is None or self._piece_masks_version != self._version:
            masks = tuple(int(value) for value in speedups.board_piece_masks(self._capsule))
            self._piece_masks_cache = masks
            self._piece_masks_version = self._version
        return self._piece_masks_cache

    def _legal_moves_tuple(self):
        if self._legal_moves_cache is None or self._legal_moves_version != self._version:
            packed_moves = speedups.board_legal_moves(self._capsule)
            self._legal_moves_cache = tuple(NativeMove(int(value)) for value in packed_moves)
            self._legal_moves_version = self._version
        return self._legal_moves_cache

    def copy(self, stack=False):
        return NativeBoard(_capsule=speedups.board_clone(self._capsule, stack))

    def push(self, move):
        speedups.board_push(self._capsule, _coerce_packed_move(move))
        self._invalidate()

    def pop(self):
        move = NativeMove(int(speedups.board_pop(self._capsule)))
        self._invalidate()
        return move

    @property
    def turn(self):
        return bool(speedups.board_turn(self._capsule))

    @property
    def fullmove_number(self):
        return int(speedups.board_fullmove_number(self._capsule))

    @property
    def halfmove_clock(self):
        return int(speedups.board_halfmove_clock(self._capsule))

    @property
    def ep_square(self):
        value = speedups.board_ep_square(self._capsule)
        return None if value is None else int(value)

    @property
    def castling_rights(self):
        return int(speedups.board_castling_rights(self._capsule))

    @property
    def move_stack(self):
        return tuple(NativeMove(int(value)) for value in speedups.board_move_stack(self._capsule))

    @property
    def pawns(self):
        return self._piece_masks()[0]

    @property
    def knights(self):
        return self._piece_masks()[1]

    @property
    def bishops(self):
        return self._piece_masks()[2]

    @property
    def rooks(self):
        return self._piece_masks()[3]

    @property
    def queens(self):
        return self._piece_masks()[4]

    @property
    def kings(self):
        return self._piece_masks()[5]

    @property
    def occupied_co(self):
        masks = self._piece_masks()
        return (masks[6], masks[7])

    def has_kingside_castling_rights(self, color):
        if color == WHITE:
            return bool(self.castling_rights & CASTLE_WHITE_KINGSIDE)
        return bool(self.castling_rights & CASTLE_BLACK_KINGSIDE)

    def has_queenside_castling_rights(self, color):
        if color == WHITE:
            return bool(self.castling_rights & CASTLE_WHITE_QUEENSIDE)
        return bool(self.castling_rights & CASTLE_BLACK_QUEENSIDE)

    def pieces(self, piece_type, color):
        if piece_type == PAWN:
            bitboard = self.pawns
        elif piece_type == KNIGHT:
            bitboard = self.knights
        elif piece_type == BISHOP:
            bitboard = self.bishops
        elif piece_type == ROOK:
            bitboard = self.rooks
        elif piece_type == QUEEN:
            bitboard = self.queens
        elif piece_type == KING:
            bitboard = self.kings
        else:
            return ()
        bitboard &= self.occupied_co[color]
        squares = []
        while bitboard:
            lsb = bitboard & -bitboard
            squares.append(lsb.bit_length() - 1)
            bitboard ^= lsb
        return tuple(squares)

    def zobrist_hash(self):
        return int(speedups.board_zobrist_hash(self._capsule))

    def is_game_over(self, claim_draw=True):
        return bool(speedups.board_is_game_over(self._capsule, bool(claim_draw)))

    def is_checkmate(self):
        return bool(speedups.board_is_checkmate(self._capsule))

    def result(self, claim_draw=True):
        return str(speedups.board_result(self._capsule, bool(claim_draw)))

    def can_claim_threefold_repetition(self):
        return bool(speedups.board_can_claim_threefold_repetition(self._capsule))

    def can_claim_fifty_moves(self):
        return bool(speedups.board_can_claim_fifty_moves(self._capsule))

    def is_repetition(self, count=2):
        return bool(speedups.board_is_repetition(self._capsule, int(count)))

    def fen(self):
        return str(speedups.board_fen(self._capsule))

    def __repr__(self):
        return f"NativeBoard({self.fen()!r})"


def make_board(fen=None):
    return NativeBoard(fen=fen)
