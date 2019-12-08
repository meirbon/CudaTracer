#include <cfenv>

// floating point error catching code
#if defined(__APPLE__) || defined(WIN32D)

static int feenableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
                 old_excepts; // previous masks

    if (fegetenv(&fenv))
        return -1;
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // unmask
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr &= ~(new_excepts << 7);

    return (fesetenv(&fenv) ? -1 : old_excepts);
}

static int fedisableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT,
                 old_excepts; // all previous masks

    if (fegetenv(&fenv))
        return -1;
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // mask
    fenv.__control |= new_excepts;
    fenv.__mxcsr |= new_excepts << 7;

    return (fesetenv(&fenv) ? -1 : old_excepts);
}

#endif

#define ENABLE_FLOAT_EXCEPTIONS feenableexcept(FE_DIVBYZERO);
#define DISABLE_FLOAT_EXCEPTIONS fedisableexcept(FE_DIVBYZERO);
#define BADFLOAT(x) ((*(uint *)&x & 0x7f000000) == 0x7f000000)