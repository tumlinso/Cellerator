set pagination off
set confirm off
set breakpoint pending on
set print thread-events off
set follow-fork-mode child
set detach-on-fork off
set width 0
handle SIGPIPE nostop noprint pass
catch signal SIGABRT
catch signal SIGBUS
catch signal SIGFPE
catch signal SIGILL
catch signal SIGSEGV
run
echo \n=== inferiors ===\n
info inferiors
echo \n=== backtrace ===\n
bt 12
echo \n=== frame 0 ===\n
frame 0
echo \n=== args ===\n
info args
echo \n=== locals ===\n
info locals
echo \n=== threads ===\n
info threads
echo \n=== short thread backtraces ===\n
thread apply all bt 4
quit
