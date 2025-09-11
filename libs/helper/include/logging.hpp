#pragma once
#include <cstdio>

#ifdef ENABLE_DEBUG_LOG

#define DBG_PRINT(fmt, ...) std::printf(fmt, ##__VA_ARGS__)


#define DBG_IF(...)        do { __VA_ARGS__ } while (0)

#define DBG_RANK_PRINT(rank, only_rank, fmt, ...) \
    do { if ((rank) == (only_rank)) DBG_PRINT(fmt, ##__VA_ARGS__); } while (0)

#else

#define DBG_PRINT(...)     do { } while (0)
#define DBG_IF(...)        do { } while (0)
#define DBG_RANK_PRINT(...) do { } while (0)

#endif

