#!/usr/bin/env python3
import sys
import termios
import tty
import subprocess

# ===== COLORS =====
RESET = "\033[0m"
BOLD  = "\033[1m"

D_CYAN   = "\033[36m"
D_GREEN  = "\033[32m"
D_YELLOW = "\033[33m"
D_RED    = "\033[31m"
D_GRAY   = "\033[90m"

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def run_cmd(cmd_list):
    print(f"{BOLD}{D_GRAY}================ COMMAND ================{RESET}")
    print(f"{BOLD}{D_GREEN}{' '.join(cmd_list)}{RESET}")
    print(f"{BOLD}{D_GRAY}================ OUTPUT ================={RESET}")
    res = subprocess.run(cmd_list)
    print(f"{BOLD}{D_GRAY}========================================{RESET}")
    return res.returncode

print(f"\n{BOLD}{D_CYAN}=== GIT HELPER ==={RESET}")
print(
    f"  {BOLD}{D_CYAN}0{RESET}:add+commit+push\n"
    f"  {BOLD}{D_CYAN}1{RESET}:status  "
    f"{BOLD}{D_CYAN}2{RESET}:add  "
    f"{BOLD}{D_CYAN}3{RESET}:commit  "
    f"{BOLD}{D_CYAN}4{RESET}:push  "
    f"{BOLD}{D_CYAN}5{RESET}:pull  "
    f"{BOLD}{D_CYAN}6{RESET}:gh login  "
    f"{BOLD}{D_CYAN}7{RESET}:gh logout\n"
    f"     or   {BOLD}{D_RED}q{RESET} to quit"
)

print(f"{BOLD}{D_YELLOW}key:{RESET} ", end="", flush=True)
key = getch()
print(key)

if key == "0":
    # add
    rc = run_cmd(["git", "add", "."])
    if rc != 0:
        sys.exit(rc)

    # commit (prompt)
    print(f"{BOLD}{D_YELLOW}message (Enter=auto):{RESET}", end=" ")
    msg = input().strip()
    if not msg:
        msg = "auto commit"
    rc = run_cmd(["git", "commit", "-m", msg])
    if rc != 0:
        sys.exit(rc)

    # push
    rc = run_cmd(["git", "push"])
    sys.exit(rc)

elif key == "1":
    sys.exit(run_cmd(["git", "status"]))

elif key == "2":
    sys.exit(run_cmd(["git", "add", "."]))

elif key == "3":
    print(f"{BOLD}{D_YELLOW}message (Enter=auto):{RESET}", end=" ")
    msg = input().strip()
    if not msg:
        msg = "auto commit"
    sys.exit(run_cmd(["git", "commit", "-m", msg]))

elif key == "4":
    sys.exit(run_cmd(["git", "push"]))

elif key == "5":
    sys.exit(run_cmd(["git", "pull"]))

elif key == "6":
    sys.exit(run_cmd(["gh", "auth", "login"]))

elif key == "7":
    sys.exit(run_cmd(["gh", "auth", "logout"]))

elif key == "q":
    print(f"{BOLD}{D_RED}exit{RESET}")
    sys.exit(0)

else:
    print(f"{BOLD}{D_RED}invalid key{RESET}")
    sys.exit(1)

