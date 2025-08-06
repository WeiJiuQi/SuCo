#pragma once
#include <iostream>
#include <sys/sysinfo.h>
#include <mutex>
#include <atomic>
#include <signal.h>
#include <unistd.h>

using namespace std;

// struct hash_pair { 
//     template <class T1, class T2> 
//     size_t operator()(const std::pair<T1, T2>& p) const
//     { 
//         auto hash1 = std::hash<T1>{}(p.first); 
//         auto hash2 = std::hash<T2>{}(p.second); 
//         return hash1 ^ hash2; 
//     } 
// }; 

template <typename T>
inline void hash_combine(std::size_t &seed, const T &val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
// auxiliary generic functions to create a hash value using a seed
template <typename T> inline void hash_val(std::size_t &seed, const T &val) {
    hash_combine(seed, val);
}
template <typename T, typename... Types>
inline void hash_val(std::size_t &seed, const T &val, const Types &... args) {
    hash_combine(seed, val);
    hash_val(seed, args...);
}

template <typename... Types>
inline std::size_t hash_val(const Types &... args) {
    std::size_t seed = 0;
    hash_val(seed, args...);
    return seed;
}

struct hash_pair {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        return hash_val(p.first, p.second);
    }
};

static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

class progress_display
{
public:
    explicit progress_display(
        unsigned long expected_count,
        std::ostream& os = std::cout,
        const std::string& s1 = "\n",
        const std::string& s2 = "",
        const std::string& s3 = "")
        : m_os(os), m_s1(s1), m_s2(s2), m_s3(s3)
    {
        restart(expected_count);
    }
    void restart(unsigned long expected_count)
    {
        //_count = _next_tic_count = _tic = 0;
        _expected_count = expected_count;
        m_os << m_s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
            << m_s2 << "|----|----|----|----|----|----|----|----|----|----|"
            << std::endl
            << m_s3;
        if (!_expected_count)
        {
            _expected_count = 1;
        }
    }
    unsigned long operator += (unsigned long increment)
    {
        std::unique_lock<std::mutex> lock(mtx);
        if ((_count += increment) >= _next_tic_count)
        {
            display_tic();
        }
        return _count;
    }
    unsigned long  operator ++ ()
    {
        return operator += (1);
    }

    //unsigned long  operator + (int x)
    //{
    //	return operator += (x);
    //}

    unsigned long count() const
    {
        return _count;
    }
    unsigned long expected_count() const
    {
        return _expected_count;
    }
private:
    std::ostream& m_os;
    const std::string m_s1;
    const std::string m_s2;
    const std::string m_s3;
    std::mutex mtx;
    std::atomic<size_t> _count{ 0 }, _expected_count{ 0 }, _next_tic_count{ 0 };
    std::atomic<unsigned> _tic{ 0 };
    void display_tic()
    {
        unsigned tics_needed = unsigned((double(_count) / _expected_count) * 50.0);
        do
        {
            m_os << '*' << std::flush;
        } while (++_tic < tics_needed);
        _next_tic_count = unsigned((_tic / 50.0) * _expected_count);
        if (_count == _expected_count)
        {
            if (_tic < 51) m_os << '*';
            m_os << std::endl;
        }
    }
};
