extern "C" float cellerator_region(float x, void(*hook)(int)) noexcept { if(hook){hook(0);hook(1);} return x*9.000000f; }
