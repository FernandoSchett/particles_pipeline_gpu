#pragma once
#include <cstdio>

#ifdef ENABLE_DEBUG_LOG

#define DBG_IF(code_block) \
    do                     \
    {                      \
        code_block         \
    } while (0)

#define DBG_PRINT(fmt, ...) \
    std::printf(fmt, ##__VA_ARGS__)

#define DBG_RANK_PRINT(rank, only_rank, fmt, ...) \
    do                                            \
    {                                             \
        if ((rank) == (only_rank))                \
            DBG_PRINT(fmt, ##__VA_ARGS__);        \
    } while (0)

#else

#define DBG_PRINT(fmt, ...) \
    do                      \
    {                       \
    } while (0)

#define DBG_RANK_PRINT(rank, only_rank, fmt, ...) \
    do                                            \
    {                                             \
    } while (0)

#define DBG_IF(code_block) \
    do                     \
    {                      \
    } while (0)
#endif
