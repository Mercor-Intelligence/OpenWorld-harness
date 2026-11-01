# Kill every process left in a timed-out bash command's session.
#
# Prefixed by the harness with: fg_dir=...; token=...;
# The command wrapper (tools.foreground_command) runs the model's command in
# its own session and writes the session id to $fg_dir/$token.sid, and, when
# the command itself finishes, its exit code to $fg_dir/$token.rc. A backend
# timeout kills only the exec's top process, so the rest of the session
# (children, `timeout N ...` wrappers, background jobs) keeps running until
# this script ends it. Sessions survive setpgid, so `timeout`'s own process
# group is still found. Reads /proc only: slim images have no ps/pkill.
#
# Prints:  sid=<sid> | sid=none (no session file, or an unusable one)
#          killed=<n>
#          rc=<exit code of the command itself, empty when it never finished>
#          survivor=<pid> <state> <comm>   (one line per process still there)

# This runs as root, but the files are written by the command's own user in a
# directory that user can write. Only processes of the .sid file's owner are
# touched, so a forged session id cannot make root kill another user's
# processes (the watchdog, the timer, the harness's own commands).
owner=
if [ -f "$fg_dir/$token.sid" ] && [ ! -L "$fg_dir/$token.sid" ]; then
    # shellcheck disable=SC2046
    set -- $(ls -ln "$fg_dir/$token.sid" 2>/dev/null)
    owner=${3:-}
fi
sid=$(cat "$fg_dir/$token.sid" 2>/dev/null)
rc=$(cat "$fg_dir/$token.rc" 2>/dev/null)
rm -f "$fg_dir/$token.sid" "$fg_dir/$token.rc" 2>/dev/null
for value in "$sid" "$owner"; do
    case "$value" in
        ''|*[!0-9]*) echo "sid=none"; echo "killed=0"; echo "rc=$rc"; exit 0 ;;
    esac
done
if [ "$sid" -le 1 ]; then
    echo "sid=none"; echo "killed=0"; echo "rc=$rc"; exit 0
fi
echo "sid=$sid"

# One pass: SIGKILL every live process whose session (stat field 6) is $sid.
# PID 1 and this shell are never touched; zombies are skipped (PID 1 may
# never reap them, so they would otherwise look like survivors).
sweep() {
    found=0
    for d in /proc/[0-9]*; do
        pid=${d#/proc/}
        [ "$pid" = 1 ] && continue
        [ "$pid" = "$$" ] && continue
        line=
        read -r line < "$d/stat" 2>/dev/null || continue
        rest=${line##*) }
        # shellcheck disable=SC2086
        set -- $rest
        [ "${4:-}" = "$sid" ] || continue
        case "$1" in Z|X|x) continue ;; esac
        state=$1
        uid=
        while read -r key real _; do
            if [ "$key" = "Uid:" ]; then uid=$real; break; fi
        done < "$d/status" 2>/dev/null
        [ "$uid" = "$owner" ] || continue
        if [ "$mode" = kill ]; then
            if kill -KILL "$pid" 2>/dev/null; then
                case " $seen " in
                    *" $pid "*) ;;
                    *) killed=$((killed + 1)); seen="$seen $pid" ;;
                esac
            fi
        else
            comm=${line#*(}
            comm=${comm%)*}
            echo "survivor=$pid $state $comm"
        fi
        found=1
    done
}

killed=0
seen=
mode=kill
round=0
while [ "$round" -lt 5 ]; do
    round=$((round + 1))
    sweep
    [ "$found" = 0 ] && break
    sleep 0.2 2>/dev/null || sleep 1
done
[ "$found" = 1 ] && { sleep 0.5 2>/dev/null || sleep 1; }
echo "killed=$killed"
echo "rc=$rc"
mode=list
sweep
exit 0
