#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace teenyzero_native {

namespace {

constexpr const char* kBoardCapsuleName = "teenyzero.native.Board";

constexpr int kEmpty = 0;
constexpr int kPawn = 1;
constexpr int kKnight = 2;
constexpr int kBishop = 3;
constexpr int kRook = 4;
constexpr int kQueen = 5;
constexpr int kKing = 6;

constexpr int kBlack = 0;
constexpr int kWhite = 1;

constexpr int kCastleWhiteKingside = 1;
constexpr int kCastleWhiteQueenside = 2;
constexpr int kCastleBlackKingside = 4;
constexpr int kCastleBlackQueenside = 8;

constexpr int kPromotionMask = 7;

struct UndoState {
    int packed_move = 0;
    int moved_piece = 0;
    int captured_piece = 0;
    int captured_square = -1;
    int rook_from = -1;
    int rook_to = -1;
    int previous_castling_rights = 0;
    int previous_ep_square = -1;
    int previous_halfmove_clock = 0;
    int previous_fullmove_number = 1;
    bool was_en_passant = false;
    bool was_castle = false;
};

struct NativeBoardState {
    std::array<int8_t, 64> squares{};
    std::uint64_t piece_bitboards[2][6]{};
    std::uint64_t occupancy[2]{};
    bool white_to_move = true;
    int castling_rights = 0;
    int ep_square = -1;
    int halfmove_clock = 0;
    int fullmove_number = 1;
    std::uint64_t zobrist_hash = 0;
    std::vector<UndoState> history;
    std::vector<int> move_stack;
    std::vector<std::uint64_t> hash_history;
};

std::uint64_t g_piece_hash[2][6][64];
std::uint64_t g_castle_hash[16];
std::uint64_t g_ep_hash[65];
std::uint64_t g_turn_hash = 0;
bool g_zobrist_initialized = false;

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

void init_zobrist() {
    if (g_zobrist_initialized) {
        return;
    }
    std::uint64_t seed = 0x123456789abcdef0ULL;
    for (int color = 0; color < 2; ++color) {
        for (int piece = 0; piece < 6; ++piece) {
            for (int square = 0; square < 64; ++square) {
                g_piece_hash[color][piece][square] = splitmix64(seed);
            }
        }
    }
    for (auto& value : g_castle_hash) {
        value = splitmix64(seed);
    }
    for (auto& value : g_ep_hash) {
        value = splitmix64(seed);
    }
    g_turn_hash = splitmix64(seed);
    g_zobrist_initialized = true;
}

inline int file_of(int square) {
    return square & 7;
}

inline int rank_of(int square) {
    return square >> 3;
}

inline std::uint64_t square_bit(int square) {
    return 1ULL << square;
}

inline int color_of(int piece) {
    return piece > 0 ? kWhite : kBlack;
}

inline int piece_type_of(int piece) {
    return std::abs(piece);
}

inline int make_piece(int color, int piece_type) {
    return color == kWhite ? piece_type : -piece_type;
}

inline int other_color(int color) {
    return color == kWhite ? kBlack : kWhite;
}

inline int move_from_square(int packed_move) {
    return packed_move & 63;
}

inline int move_to_square(int packed_move) {
    return (packed_move >> 6) & 63;
}

inline int move_promotion(int packed_move) {
    return (packed_move >> 12) & kPromotionMask;
}

inline int pack_move(int from_square, int to_square, int promotion = 0) {
    return from_square | (to_square << 6) | (promotion << 12);
}

void clear_bitboards(NativeBoardState& board) {
    for (auto& side : board.piece_bitboards) {
        for (auto& mask : side) {
            mask = 0;
        }
    }
    board.occupancy[0] = 0;
    board.occupancy[1] = 0;
}

void add_piece(NativeBoardState& board, int square, int piece) {
    board.squares[static_cast<std::size_t>(square)] = static_cast<int8_t>(piece);
    if (piece == kEmpty) {
        return;
    }
    const int color = color_of(piece);
    const int piece_type = piece_type_of(piece) - 1;
    const std::uint64_t bit = square_bit(square);
    board.piece_bitboards[color][piece_type] |= bit;
    board.occupancy[color] |= bit;
}

void remove_piece(NativeBoardState& board, int square) {
    const int piece = board.squares[static_cast<std::size_t>(square)];
    if (piece == kEmpty) {
        return;
    }
    const int color = color_of(piece);
    const int piece_type = piece_type_of(piece) - 1;
    const std::uint64_t bit = square_bit(square);
    board.piece_bitboards[color][piece_type] &= ~bit;
    board.occupancy[color] &= ~bit;
    board.squares[static_cast<std::size_t>(square)] = 0;
}

void move_piece(NativeBoardState& board, int from_square, int to_square) {
    const int piece = board.squares[static_cast<std::size_t>(from_square)];
    remove_piece(board, from_square);
    add_piece(board, to_square, piece);
}

std::uint64_t compute_hash(const NativeBoardState& board) {
    init_zobrist();
    std::uint64_t hash = 0;
    for (int square = 0; square < 64; ++square) {
        const int piece = board.squares[static_cast<std::size_t>(square)];
        if (piece == kEmpty) {
            continue;
        }
        const int color = color_of(piece);
        const int piece_type = piece_type_of(piece) - 1;
        hash ^= g_piece_hash[color][piece_type][square];
    }
    hash ^= g_castle_hash[board.castling_rights & 15];
    hash ^= g_ep_hash[board.ep_square >= 0 ? board.ep_square : 64];
    if (board.white_to_move) {
        hash ^= g_turn_hash;
    }
    return hash;
}

void refresh_hash(NativeBoardState& board) {
    board.zobrist_hash = compute_hash(board);
}

int parse_square(const std::string& token) {
    if (token.size() != 2) {
        return -1;
    }
    const char file_char = token[0];
    const char rank_char = token[1];
    if (file_char < 'a' || file_char > 'h' || rank_char < '1' || rank_char > '8') {
        return -1;
    }
    return (rank_char - '1') * 8 + (file_char - 'a');
}

std::string square_to_string(int square) {
    if (square < 0 || square >= 64) {
        return "-";
    }
    std::string out(2, ' ');
    out[0] = static_cast<char>('a' + file_of(square));
    out[1] = static_cast<char>('1' + rank_of(square));
    return out;
}

int piece_from_fen_char(char symbol) {
    const bool is_upper = symbol >= 'A' && symbol <= 'Z';
    const char lower = static_cast<char>(std::tolower(symbol));
    int piece_type = 0;
    switch (lower) {
        case 'p':
            piece_type = kPawn;
            break;
        case 'n':
            piece_type = kKnight;
            break;
        case 'b':
            piece_type = kBishop;
            break;
        case 'r':
            piece_type = kRook;
            break;
        case 'q':
            piece_type = kQueen;
            break;
        case 'k':
            piece_type = kKing;
            break;
        default:
            return 0;
    }
    return is_upper ? piece_type : -piece_type;
}

char fen_char_from_piece(int piece) {
    const int piece_type = piece_type_of(piece);
    char out = ' ';
    switch (piece_type) {
        case kPawn:
            out = 'p';
            break;
        case kKnight:
            out = 'n';
            break;
        case kBishop:
            out = 'b';
            break;
        case kRook:
            out = 'r';
            break;
        case kQueen:
            out = 'q';
            break;
        case kKing:
            out = 'k';
            break;
        default:
            out = '?';
            break;
    }
    if (piece > 0) {
        out = static_cast<char>(std::toupper(out));
    }
    return out;
}

bool load_fen(NativeBoardState& board, const std::string& fen) {
    clear_bitboards(board);
    board.squares.fill(0);
    board.history.clear();
    board.move_stack.clear();
    board.hash_history.clear();
    board.white_to_move = true;
    board.castling_rights = 0;
    board.ep_square = -1;
    board.halfmove_clock = 0;
    board.fullmove_number = 1;

    std::istringstream stream(fen);
    std::string placement;
    std::string active_color;
    std::string castling;
    std::string ep_target;
    if (!(stream >> placement >> active_color >> castling >> ep_target >> board.halfmove_clock >> board.fullmove_number)) {
        return false;
    }

    int rank = 7;
    int file = 0;
    for (char symbol : placement) {
        if (symbol == '/') {
            --rank;
            file = 0;
            continue;
        }
        if (symbol >= '1' && symbol <= '8') {
            file += symbol - '0';
            continue;
        }
        if (rank < 0 || file > 7) {
            return false;
        }
        const int piece = piece_from_fen_char(symbol);
        if (piece == 0) {
            return false;
        }
        add_piece(board, rank * 8 + file, piece);
        ++file;
    }

    board.white_to_move = active_color == "w";
    if (castling.find('K') != std::string::npos) {
        board.castling_rights |= kCastleWhiteKingside;
    }
    if (castling.find('Q') != std::string::npos) {
        board.castling_rights |= kCastleWhiteQueenside;
    }
    if (castling.find('k') != std::string::npos) {
        board.castling_rights |= kCastleBlackKingside;
    }
    if (castling.find('q') != std::string::npos) {
        board.castling_rights |= kCastleBlackQueenside;
    }
    board.ep_square = ep_target == "-" ? -1 : parse_square(ep_target);
    refresh_hash(board);
    board.hash_history.push_back(board.zobrist_hash);
    return true;
}

NativeBoardState make_initial_board() {
    NativeBoardState board;
    load_fen(board, "rn1qkbnr/pppbpppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    // Replace with the actual standard chess start position.
    load_fen(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    return board;
}

NativeBoardState* clone_board(const NativeBoardState& source, int stack_limit) {
    auto* copy = new NativeBoardState(source);
    if (stack_limit >= 0 && static_cast<std::size_t>(stack_limit) < copy->history.size()) {
        const auto keep = static_cast<std::size_t>(stack_limit);
        copy->history.erase(copy->history.begin(), copy->history.end() - static_cast<std::ptrdiff_t>(keep));
        copy->move_stack.erase(copy->move_stack.begin(), copy->move_stack.end() - static_cast<std::ptrdiff_t>(keep));
    }
    return copy;
}

bool is_on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

int king_square(const NativeBoardState& board, int color) {
    std::uint64_t kings = board.piece_bitboards[color][kKing - 1];
    if (kings == 0) {
        return -1;
    }
    return __builtin_ctzll(kings);
}

bool square_attacked_by(const NativeBoardState& board, int square, int attacker_color) {
    const int file = file_of(square);
    const int rank = rank_of(square);

    const int pawn_step = attacker_color == kWhite ? -1 : 1;
    for (int df : {-1, 1}) {
        const int src_file = file + df;
        const int src_rank = rank + pawn_step;
        if (!is_on_board(src_file, src_rank)) {
            continue;
        }
        const int src_square = src_rank * 8 + src_file;
        if (board.squares[static_cast<std::size_t>(src_square)] == make_piece(attacker_color, kPawn)) {
            return true;
        }
    }

    constexpr std::array<std::pair<int, int>, 8> knight_offsets = {{
        {1, 2}, {2, 1}, {2, -1}, {1, -2},
        {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
    }};
    for (const auto& [df, dr] : knight_offsets) {
        const int src_file = file + df;
        const int src_rank = rank + dr;
        if (!is_on_board(src_file, src_rank)) {
            continue;
        }
        const int src_square = src_rank * 8 + src_file;
        if (board.squares[static_cast<std::size_t>(src_square)] == make_piece(attacker_color, kKnight)) {
            return true;
        }
    }

    constexpr std::array<std::pair<int, int>, 4> bishop_dirs = {{
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
    }};
    for (const auto& [df, dr] : bishop_dirs) {
        int cursor_file = file + df;
        int cursor_rank = rank + dr;
        while (is_on_board(cursor_file, cursor_rank)) {
            const int cursor_square = cursor_rank * 8 + cursor_file;
            const int piece = board.squares[static_cast<std::size_t>(cursor_square)];
            if (piece != kEmpty) {
                if (piece == make_piece(attacker_color, kBishop) || piece == make_piece(attacker_color, kQueen)) {
                    return true;
                }
                break;
            }
            cursor_file += df;
            cursor_rank += dr;
        }
    }

    constexpr std::array<std::pair<int, int>, 4> rook_dirs = {{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    }};
    for (const auto& [df, dr] : rook_dirs) {
        int cursor_file = file + df;
        int cursor_rank = rank + dr;
        while (is_on_board(cursor_file, cursor_rank)) {
            const int cursor_square = cursor_rank * 8 + cursor_file;
            const int piece = board.squares[static_cast<std::size_t>(cursor_square)];
            if (piece != kEmpty) {
                if (piece == make_piece(attacker_color, kRook) || piece == make_piece(attacker_color, kQueen)) {
                    return true;
                }
                break;
            }
            cursor_file += df;
            cursor_rank += dr;
        }
    }

    constexpr std::array<std::pair<int, int>, 8> king_offsets = {{
        {1, 1}, {1, 0}, {1, -1}, {0, 1},
        {0, -1}, {-1, 1}, {-1, 0}, {-1, -1},
    }};
    for (const auto& [df, dr] : king_offsets) {
        const int src_file = file + df;
        const int src_rank = rank + dr;
        if (!is_on_board(src_file, src_rank)) {
            continue;
        }
        const int src_square = src_rank * 8 + src_file;
        if (board.squares[static_cast<std::size_t>(src_square)] == make_piece(attacker_color, kKing)) {
            return true;
        }
    }

    return false;
}

bool in_check(const NativeBoardState& board, int color) {
    const int king_sq = king_square(board, color);
    if (king_sq < 0) {
        return false;
    }
    return square_attacked_by(board, king_sq, other_color(color));
}

void update_castling_rights_for_square(NativeBoardState& board, int square) {
    switch (square) {
        case 0:
            board.castling_rights &= ~kCastleWhiteQueenside;
            break;
        case 4:
            board.castling_rights &= ~(kCastleWhiteKingside | kCastleWhiteQueenside);
            break;
        case 7:
            board.castling_rights &= ~kCastleWhiteKingside;
            break;
        case 56:
            board.castling_rights &= ~kCastleBlackQueenside;
            break;
        case 60:
            board.castling_rights &= ~(kCastleBlackKingside | kCastleBlackQueenside);
            break;
        case 63:
            board.castling_rights &= ~kCastleBlackKingside;
            break;
        default:
            break;
    }
}

bool push_unchecked(NativeBoardState& board, int packed_move) {
    const int from_square = move_from_square(packed_move);
    const int to_square = move_to_square(packed_move);
    const int promotion = move_promotion(packed_move);
    if (from_square < 0 || from_square >= 64 || to_square < 0 || to_square >= 64) {
        return false;
    }
    const int moved_piece = board.squares[static_cast<std::size_t>(from_square)];
    if (moved_piece == kEmpty) {
        return false;
    }

    UndoState undo;
    undo.packed_move = packed_move;
    undo.moved_piece = moved_piece;
    undo.previous_castling_rights = board.castling_rights;
    undo.previous_ep_square = board.ep_square;
    undo.previous_halfmove_clock = board.halfmove_clock;
    undo.previous_fullmove_number = board.fullmove_number;

    const int moving_color = color_of(moved_piece);
    const int target_piece = board.squares[static_cast<std::size_t>(to_square)];
    undo.captured_piece = target_piece;
    undo.captured_square = to_square;

    board.ep_square = -1;
    if (piece_type_of(moved_piece) == kPawn || target_piece != kEmpty) {
        board.halfmove_clock = 0;
    } else {
        board.halfmove_clock += 1;
    }

    if (piece_type_of(moved_piece) == kPawn && to_square == undo.previous_ep_square && target_piece == kEmpty) {
        undo.was_en_passant = true;
        undo.captured_square = moving_color == kWhite ? to_square - 8 : to_square + 8;
        undo.captured_piece = board.squares[static_cast<std::size_t>(undo.captured_square)];
        remove_piece(board, undo.captured_square);
        board.halfmove_clock = 0;
    } else if (target_piece != kEmpty) {
        remove_piece(board, to_square);
    }

    update_castling_rights_for_square(board, from_square);
    if (undo.captured_piece != kEmpty) {
        update_castling_rights_for_square(board, undo.captured_square);
    }

    if (piece_type_of(moved_piece) == kKing && std::abs(to_square - from_square) == 2) {
        undo.was_castle = true;
        if (to_square > from_square) {
            undo.rook_from = moving_color == kWhite ? 7 : 63;
            undo.rook_to = to_square - 1;
        } else {
            undo.rook_from = moving_color == kWhite ? 0 : 56;
            undo.rook_to = to_square + 1;
        }
        move_piece(board, undo.rook_from, undo.rook_to);
    }

    remove_piece(board, from_square);
    int placed_piece = moved_piece;
    if (promotion != 0 && piece_type_of(moved_piece) == kPawn) {
        placed_piece = make_piece(moving_color, promotion);
    }
    add_piece(board, to_square, placed_piece);

    if (piece_type_of(moved_piece) == kPawn && std::abs(to_square - from_square) == 16) {
        board.ep_square = moving_color == kWhite ? from_square + 8 : from_square - 8;
    }

    if (moving_color == kBlack) {
        board.fullmove_number += 1;
    }

    board.white_to_move = !board.white_to_move;
    refresh_hash(board);
    board.history.push_back(undo);
    board.move_stack.push_back(packed_move);
    board.hash_history.push_back(board.zobrist_hash);
    return true;
}

bool pop_last_move(NativeBoardState& board) {
    if (board.history.empty()) {
        return false;
    }

    const UndoState undo = board.history.back();
    board.history.pop_back();
    if (!board.move_stack.empty()) {
        board.move_stack.pop_back();
    }
    if (!board.hash_history.empty()) {
        board.hash_history.pop_back();
    }

    const int from_square = move_from_square(undo.packed_move);
    const int to_square = move_to_square(undo.packed_move);

    board.white_to_move = color_of(undo.moved_piece) == kWhite;
    board.castling_rights = undo.previous_castling_rights;
    board.ep_square = undo.previous_ep_square;
    board.halfmove_clock = undo.previous_halfmove_clock;
    board.fullmove_number = undo.previous_fullmove_number;

    if (undo.was_castle) {
        move_piece(board, undo.rook_to, undo.rook_from);
    }

    remove_piece(board, to_square);
    add_piece(board, from_square, undo.moved_piece);
    if (undo.captured_piece != kEmpty) {
        add_piece(board, undo.captured_square, undo.captured_piece);
    }

    refresh_hash(board);
    return true;
}

void append_pawn_moves(const NativeBoardState& board, int color, std::vector<int>& out_moves) {
    const int direction = color == kWhite ? 8 : -8;
    const int double_step_rank = color == kWhite ? 1 : 6;
    const int promotion_rank = color == kWhite ? 6 : 1;
    std::uint64_t pawns = board.piece_bitboards[color][kPawn - 1];

    while (pawns) {
        const int square = __builtin_ctzll(pawns);
        pawns &= pawns - 1;
        const int file = file_of(square);
        const int rank = rank_of(square);
        const int one_step = square + direction;
        if (one_step >= 0 && one_step < 64 && board.squares[static_cast<std::size_t>(one_step)] == kEmpty) {
            if (rank == promotion_rank) {
                for (int promotion : {kQueen, kRook, kBishop, kKnight}) {
                    out_moves.push_back(pack_move(square, one_step, promotion));
                }
            } else {
                out_moves.push_back(pack_move(square, one_step));
                if (rank == double_step_rank) {
                    const int two_step = square + direction * 2;
                    if (board.squares[static_cast<std::size_t>(two_step)] == kEmpty) {
                        out_moves.push_back(pack_move(square, two_step));
                    }
                }
            }
        }

        for (int df : {-1, 1}) {
            const int target_file = file + df;
            if (target_file < 0 || target_file > 7) {
                continue;
            }
            const int target_square = one_step + df;
            if (target_square < 0 || target_square >= 64) {
                continue;
            }
            const int target_piece = board.squares[static_cast<std::size_t>(target_square)];
            const bool is_capture = target_piece != kEmpty && color_of(target_piece) != color;
            const bool is_ep = target_square == board.ep_square;
            if (!is_capture && !is_ep) {
                continue;
            }
            if (rank == promotion_rank) {
                for (int promotion : {kQueen, kRook, kBishop, kKnight}) {
                    out_moves.push_back(pack_move(square, target_square, promotion));
                }
            } else {
                out_moves.push_back(pack_move(square, target_square));
            }
        }
    }
}

void append_knight_moves(const NativeBoardState& board, int color, std::vector<int>& out_moves) {
    constexpr std::array<std::pair<int, int>, 8> offsets = {{
        {1, 2}, {2, 1}, {2, -1}, {1, -2},
        {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
    }};
    std::uint64_t pieces = board.piece_bitboards[color][kKnight - 1];
    while (pieces) {
        const int square = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        const int file = file_of(square);
        const int rank = rank_of(square);
        for (const auto& [df, dr] : offsets) {
            const int target_file = file + df;
            const int target_rank = rank + dr;
            if (!is_on_board(target_file, target_rank)) {
                continue;
            }
            const int target_square = target_rank * 8 + target_file;
            const int target_piece = board.squares[static_cast<std::size_t>(target_square)];
            if (target_piece == kEmpty || color_of(target_piece) != color) {
                out_moves.push_back(pack_move(square, target_square));
            }
        }
    }
}

void append_slider_moves(
    const NativeBoardState& board,
    int color,
    int piece_type,
    const std::array<std::pair<int, int>, 4>& directions,
    std::vector<int>& out_moves
) {
    std::uint64_t pieces = board.piece_bitboards[color][piece_type - 1];
    while (pieces) {
        const int square = __builtin_ctzll(pieces);
        pieces &= pieces - 1;
        const int file = file_of(square);
        const int rank = rank_of(square);
        for (const auto& [df, dr] : directions) {
            int target_file = file + df;
            int target_rank = rank + dr;
            while (is_on_board(target_file, target_rank)) {
                const int target_square = target_rank * 8 + target_file;
                const int target_piece = board.squares[static_cast<std::size_t>(target_square)];
                if (target_piece == kEmpty) {
                    out_moves.push_back(pack_move(square, target_square));
                } else {
                    if (color_of(target_piece) != color) {
                        out_moves.push_back(pack_move(square, target_square));
                    }
                    break;
                }
                target_file += df;
                target_rank += dr;
            }
        }
    }
}

void append_king_moves(const NativeBoardState& board, int color, std::vector<int>& out_moves) {
    constexpr std::array<std::pair<int, int>, 8> offsets = {{
        {1, 1}, {1, 0}, {1, -1}, {0, 1},
        {0, -1}, {-1, 1}, {-1, 0}, {-1, -1},
    }};

    const int square = king_square(board, color);
    if (square < 0) {
        return;
    }
    const int file = file_of(square);
    const int rank = rank_of(square);
    for (const auto& [df, dr] : offsets) {
        const int target_file = file + df;
        const int target_rank = rank + dr;
        if (!is_on_board(target_file, target_rank)) {
            continue;
        }
        const int target_square = target_rank * 8 + target_file;
        const int target_piece = board.squares[static_cast<std::size_t>(target_square)];
        if (target_piece == kEmpty || color_of(target_piece) != color) {
            out_moves.push_back(pack_move(square, target_square));
        }
    }

    if (color == kWhite && square == 4) {
        if ((board.castling_rights & kCastleWhiteKingside) != 0
            && board.squares[5] == kEmpty
            && board.squares[6] == kEmpty
            && board.squares[7] == make_piece(kWhite, kRook)
            && !square_attacked_by(board, 4, kBlack)
            && !square_attacked_by(board, 5, kBlack)
            && !square_attacked_by(board, 6, kBlack)) {
            out_moves.push_back(pack_move(4, 6));
        }
        if ((board.castling_rights & kCastleWhiteQueenside) != 0
            && board.squares[1] == kEmpty
            && board.squares[2] == kEmpty
            && board.squares[3] == kEmpty
            && board.squares[0] == make_piece(kWhite, kRook)
            && !square_attacked_by(board, 4, kBlack)
            && !square_attacked_by(board, 3, kBlack)
            && !square_attacked_by(board, 2, kBlack)) {
            out_moves.push_back(pack_move(4, 2));
        }
    } else if (color == kBlack && square == 60) {
        if ((board.castling_rights & kCastleBlackKingside) != 0
            && board.squares[61] == kEmpty
            && board.squares[62] == kEmpty
            && board.squares[63] == make_piece(kBlack, kRook)
            && !square_attacked_by(board, 60, kWhite)
            && !square_attacked_by(board, 61, kWhite)
            && !square_attacked_by(board, 62, kWhite)) {
            out_moves.push_back(pack_move(60, 62));
        }
        if ((board.castling_rights & kCastleBlackQueenside) != 0
            && board.squares[57] == kEmpty
            && board.squares[58] == kEmpty
            && board.squares[59] == kEmpty
            && board.squares[56] == make_piece(kBlack, kRook)
            && !square_attacked_by(board, 60, kWhite)
            && !square_attacked_by(board, 59, kWhite)
            && !square_attacked_by(board, 58, kWhite)) {
            out_moves.push_back(pack_move(60, 58));
        }
    }
}

std::vector<int> generate_legal_moves(NativeBoardState& board) {
    const int color = board.white_to_move ? kWhite : kBlack;
    std::vector<int> pseudo_moves;
    pseudo_moves.reserve(128);
    append_pawn_moves(board, color, pseudo_moves);
    append_knight_moves(board, color, pseudo_moves);
    append_slider_moves(
        board,
        color,
        kBishop,
        std::array<std::pair<int, int>, 4>{{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}},
        pseudo_moves
    );
    append_slider_moves(
        board,
        color,
        kRook,
        std::array<std::pair<int, int>, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}},
        pseudo_moves
    );
    append_slider_moves(
        board,
        color,
        kQueen,
        std::array<std::pair<int, int>, 4>{{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}},
        pseudo_moves
    );
    append_slider_moves(
        board,
        color,
        kQueen,
        std::array<std::pair<int, int>, 4>{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}},
        pseudo_moves
    );
    append_king_moves(board, color, pseudo_moves);

    std::vector<int> legal_moves;
    legal_moves.reserve(pseudo_moves.size());
    for (int packed_move : pseudo_moves) {
        if (!push_unchecked(board, packed_move)) {
            continue;
        }
        const bool legal = !in_check(board, color);
        pop_last_move(board);
        if (legal) {
            legal_moves.push_back(packed_move);
        }
    }
    return legal_moves;
}

int repetition_count(const NativeBoardState& board) {
    if (board.hash_history.empty()) {
        return 0;
    }
    const std::uint64_t current = board.hash_history.back();
    int count = 0;
    const std::size_t limit = std::min<std::size_t>(
        board.hash_history.size(),
        static_cast<std::size_t>(board.halfmove_clock + 1)
    );
    for (std::size_t idx = board.hash_history.size(); idx-- > board.hash_history.size() - limit;) {
        if (board.hash_history[idx] == current) {
            ++count;
        }
    }
    return count;
}

bool insufficient_material(const NativeBoardState& board) {
    const int white_non_king = __builtin_popcountll(
        board.piece_bitboards[kWhite][kPawn - 1]
        | board.piece_bitboards[kWhite][kRook - 1]
        | board.piece_bitboards[kWhite][kQueen - 1]
    );
    const int black_non_king = __builtin_popcountll(
        board.piece_bitboards[kBlack][kPawn - 1]
        | board.piece_bitboards[kBlack][kRook - 1]
        | board.piece_bitboards[kBlack][kQueen - 1]
    );
    if (white_non_king > 0 || black_non_king > 0) {
        return false;
    }
    const int white_minors = __builtin_popcountll(
        board.piece_bitboards[kWhite][kKnight - 1] | board.piece_bitboards[kWhite][kBishop - 1]
    );
    const int black_minors = __builtin_popcountll(
        board.piece_bitboards[kBlack][kKnight - 1] | board.piece_bitboards[kBlack][kBishop - 1]
    );
    return white_minors <= 1 && black_minors <= 1;
}

std::string board_result(NativeBoardState& board, bool claim_draw) {
    if (claim_draw && (board.halfmove_clock >= 100 || repetition_count(board) >= 3 || insufficient_material(board))) {
        return "1/2-1/2";
    }
    auto legal_moves = generate_legal_moves(board);
    if (!legal_moves.empty()) {
        return "*";
    }
    const int color = board.white_to_move ? kWhite : kBlack;
    if (in_check(board, color)) {
        return color == kWhite ? "0-1" : "1-0";
    }
    return "1/2-1/2";
}

bool parse_copy_stack(PyObject* stack_obj, int& out_stack_limit) {
    if (stack_obj == nullptr || stack_obj == Py_True) {
        out_stack_limit = -1;
        return true;
    }
    if (stack_obj == Py_False) {
        out_stack_limit = 0;
        return true;
    }
    const long value = PyLong_AsLong(stack_obj);
    if (value == -1 && PyErr_Occurred()) {
        return false;
    }
    out_stack_limit = value < 0 ? -1 : static_cast<int>(value);
    return true;
}

NativeBoardState* get_board_state(PyObject* capsule) {
    return reinterpret_cast<NativeBoardState*>(PyCapsule_GetPointer(capsule, kBoardCapsuleName));
}

void board_capsule_destructor(PyObject* capsule) {
    auto* board = get_board_state(capsule);
    delete board;
}

PyObject* wrap_board(NativeBoardState* board) {
    return PyCapsule_New(board, kBoardCapsuleName, board_capsule_destructor);
}

PyObject* build_move_tuple(const std::vector<int>& packed_moves) {
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(packed_moves.size()));
    if (tuple == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(packed_moves.size()); ++index) {
        PyObject* value = PyLong_FromLong(packed_moves[static_cast<std::size_t>(index)]);
        if (value == nullptr) {
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, index, value);
    }
    return tuple;
}

}  // namespace

PyObject* py_move_from_uci(PyObject*, PyObject* args) {
    const char* uci_cstr = nullptr;
    if (!PyArg_ParseTuple(args, "s", &uci_cstr)) {
        return nullptr;
    }
    const std::string uci(uci_cstr);
    if (uci.size() < 4 || uci.size() > 5) {
        PyErr_SetString(PyExc_ValueError, "invalid UCI move");
        return nullptr;
    }
    const int from_square = parse_square(uci.substr(0, 2));
    const int to_square = parse_square(uci.substr(2, 2));
    if (from_square < 0 || to_square < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid UCI square");
        return nullptr;
    }
    int promotion = 0;
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'n':
                promotion = kKnight;
                break;
            case 'b':
                promotion = kBishop;
                break;
            case 'r':
                promotion = kRook;
                break;
            case 'q':
                promotion = kQueen;
                break;
            default:
                PyErr_SetString(PyExc_ValueError, "invalid promotion piece");
                return nullptr;
        }
    }
    return PyLong_FromLong(pack_move(from_square, to_square, promotion));
}

PyObject* py_move_uci(PyObject*, PyObject* args) {
    int packed_move = 0;
    if (!PyArg_ParseTuple(args, "i", &packed_move)) {
        return nullptr;
    }
    std::string uci = square_to_string(move_from_square(packed_move)) + square_to_string(move_to_square(packed_move));
    switch (move_promotion(packed_move)) {
        case kKnight:
            uci.push_back('n');
            break;
        case kBishop:
            uci.push_back('b');
            break;
        case kRook:
            uci.push_back('r');
            break;
        case kQueen:
            uci.push_back('q');
            break;
        default:
            break;
    }
    return PyUnicode_FromStringAndSize(uci.c_str(), static_cast<Py_ssize_t>(uci.size()));
}

PyObject* py_board_new(PyObject*, PyObject* args) {
    const char* fen_cstr = nullptr;
    if (!PyArg_ParseTuple(args, "|z", &fen_cstr)) {
        return nullptr;
    }
    auto* board = new NativeBoardState(fen_cstr == nullptr ? make_initial_board() : NativeBoardState{});
    if (fen_cstr != nullptr && !load_fen(*board, fen_cstr)) {
        delete board;
        PyErr_SetString(PyExc_ValueError, "failed to parse FEN");
        return nullptr;
    }
    return wrap_board(board);
}

PyObject* py_board_clone(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* stack_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O|O", &capsule, &stack_obj)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    int stack_limit = -1;
    if (!parse_copy_stack(stack_obj, stack_limit)) {
        return nullptr;
    }
    return wrap_board(clone_board(*board, stack_limit));
}

PyObject* py_board_turn(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyBool_FromLong(board->white_to_move ? 1 : 0);
}

PyObject* py_board_fullmove_number(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyLong_FromLong(board->fullmove_number);
}

PyObject* py_board_halfmove_clock(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyLong_FromLong(board->halfmove_clock);
}

PyObject* py_board_ep_square(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    if (board->ep_square < 0) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(board->ep_square);
}

PyObject* py_board_castling_rights(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyLong_FromLong(board->castling_rights);
}

PyObject* py_board_piece_masks(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    PyObject* tuple = PyTuple_New(8);
    if (tuple == nullptr) {
        return nullptr;
    }
    const std::uint64_t masks[8] = {
        board->piece_bitboards[kWhite][kPawn - 1] | board->piece_bitboards[kBlack][kPawn - 1],
        board->piece_bitboards[kWhite][kKnight - 1] | board->piece_bitboards[kBlack][kKnight - 1],
        board->piece_bitboards[kWhite][kBishop - 1] | board->piece_bitboards[kBlack][kBishop - 1],
        board->piece_bitboards[kWhite][kRook - 1] | board->piece_bitboards[kBlack][kRook - 1],
        board->piece_bitboards[kWhite][kQueen - 1] | board->piece_bitboards[kBlack][kQueen - 1],
        board->piece_bitboards[kWhite][kKing - 1] | board->piece_bitboards[kBlack][kKing - 1],
        board->occupancy[kBlack],
        board->occupancy[kWhite],
    };
    for (Py_ssize_t index = 0; index < 8; ++index) {
        PyTuple_SET_ITEM(tuple, index, PyLong_FromUnsignedLongLong(masks[index]));
    }
    return tuple;
}

PyObject* py_board_legal_moves(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return build_move_tuple(generate_legal_moves(*board));
}

PyObject* py_board_push(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    int packed_move = 0;
    if (!PyArg_ParseTuple(args, "Oi", &capsule, &packed_move)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    if (!push_unchecked(*board, packed_move)) {
        PyErr_SetString(PyExc_ValueError, "invalid move");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* py_board_pop(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    if (board->history.empty()) {
        PyErr_SetString(PyExc_IndexError, "empty move stack");
        return nullptr;
    }
    const int packed_move = board->history.back().packed_move;
    if (!pop_last_move(*board)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to pop move");
        return nullptr;
    }
    return PyLong_FromLong(packed_move);
}

PyObject* py_board_move_stack(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return build_move_tuple(board->move_stack);
}

PyObject* py_board_is_game_over(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    int claim_draw = 1;
    if (!PyArg_ParseTuple(args, "O|p", &capsule, &claim_draw)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyBool_FromLong(board_result(*board, claim_draw != 0) != "*" ? 1 : 0);
}

PyObject* py_board_is_checkmate(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    const auto legal_moves = generate_legal_moves(*board);
    const int color = board->white_to_move ? kWhite : kBlack;
    return PyBool_FromLong(legal_moves.empty() && in_check(*board, color) ? 1 : 0);
}

PyObject* py_board_result(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    int claim_draw = 1;
    if (!PyArg_ParseTuple(args, "O|p", &capsule, &claim_draw)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    const auto result = board_result(*board, claim_draw != 0);
    return PyUnicode_FromStringAndSize(result.c_str(), static_cast<Py_ssize_t>(result.size()));
}

PyObject* py_board_can_claim_threefold_repetition(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyBool_FromLong(repetition_count(*board) >= 3 ? 1 : 0);
}

PyObject* py_board_can_claim_fifty_moves(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyBool_FromLong(board->halfmove_clock >= 100 ? 1 : 0);
}

PyObject* py_board_is_repetition(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    int count = 2;
    if (!PyArg_ParseTuple(args, "O|i", &capsule, &count)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyBool_FromLong(repetition_count(*board) >= count ? 1 : 0);
}

PyObject* py_board_zobrist_hash(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }
    return PyLong_FromUnsignedLongLong(board->zobrist_hash);
}

PyObject* py_board_fen(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    auto* board = get_board_state(capsule);
    if (board == nullptr) {
        return nullptr;
    }

    std::ostringstream fen;
    for (int rank = 7; rank >= 0; --rank) {
        int empty_run = 0;
        for (int file = 0; file < 8; ++file) {
            const int square = rank * 8 + file;
            const int piece = board->squares[static_cast<std::size_t>(square)];
            if (piece == kEmpty) {
                empty_run += 1;
                continue;
            }
            if (empty_run > 0) {
                fen << empty_run;
                empty_run = 0;
            }
            fen << fen_char_from_piece(piece);
        }
        if (empty_run > 0) {
            fen << empty_run;
        }
        if (rank > 0) {
            fen << '/';
        }
    }
    fen << ' ' << (board->white_to_move ? 'w' : 'b') << ' ';
    if (board->castling_rights == 0) {
        fen << '-';
    } else {
        if ((board->castling_rights & kCastleWhiteKingside) != 0) {
            fen << 'K';
        }
        if ((board->castling_rights & kCastleWhiteQueenside) != 0) {
            fen << 'Q';
        }
        if ((board->castling_rights & kCastleBlackKingside) != 0) {
            fen << 'k';
        }
        if ((board->castling_rights & kCastleBlackQueenside) != 0) {
            fen << 'q';
        }
    }
    fen << ' ';
    fen << (board->ep_square >= 0 ? square_to_string(board->ep_square) : "-");
    fen << ' ' << board->halfmove_clock;
    fen << ' ' << board->fullmove_number;
    const std::string out = fen.str();
    return PyUnicode_FromStringAndSize(out.c_str(), static_cast<Py_ssize_t>(out.size()));
}

}  // namespace teenyzero_native
