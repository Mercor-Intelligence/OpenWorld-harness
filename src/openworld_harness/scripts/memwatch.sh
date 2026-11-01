# Memory watchdog: enforce the sandbox's disclosed memory limit.
#
# Run as root, detached:  sh memwatch.sh STATE THRESHOLD_BYTES LIMIT_BYTES SOURCE TICKS [SCOPE_SID]
#
# Modal (and Harbor's other backends by default) treat the sandbox memory
# limit as a request: nothing stops a run at the limit, and going far over it
# lets host pressure kill the WHOLE sandbox, which ends the run. This loop
# reads the sandbox's memory use once a second and, above THRESHOLD_BYTES,
# SIGKILLs the process with the largest resident set, as the kernel's OOM
# killer would. Never PID 1, never itself, never a process with
# oom_score_adj -1000.
#
# SOURCE: auto | cgroup2 | cgroup1 | meminfo | file:<path> (tests).
#   cgroup2  memory.current - (file - shmem)
#   cgroup1  memory/memory.usage_in_bytes - (total_cache - total_shmem)
#            i.e. without page cache the kernel can reclaim (active or not),
#            but with tmpfs and shared memory, which killing could free
#   meminfo  MemTotal - MemAvailable, only under gVisor, where /proc/meminfo
#            describes the sandbox (elsewhere it describes the host)
#   auto     the first of those that is readable here, else none
# TICKS: seconds to live (a safety net; the harness stops it explicitly).
# SCOPE_SID: only processes of this session are candidates (tests).
#
# Files in STATE (root-only):
#   pid          this loop's PID (one instance per STATE)
#   status       "source=<s> used=<bytes> page=<bytes>" once started, or source=none
#   events.tsv   seq, epoch, kind, used, limit, pid, rss, command (tab-separated)
#                kind: kill | nokill (over the threshold, no process big enough) | expire
#
# Builtins only on the per-second path (slim images have no ps/pkill; mawk
# aborts on vanished /proc files).

state=$1
threshold=$2
limit=$3
src=$4
ticks=$5
scope=${6:-}

case "$threshold$limit$ticks" in
    ''|*[!0-9]*) echo "memwatch: bad arguments" >&2; exit 2 ;;
esac
[ -d "$state" ] || mkdir -p "$state" || exit 2

# One instance per state directory.
old=
{ read -r old < "$state/pid"; } 2>/dev/null
if [ -n "$old" ] && [ "$old" != "$$" ] && [ -r "/proc/$old/cmdline" ]; then
    oldcmd=$(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null)
    case "$oldcmd" in
        *memwatch.sh*) exit 0 ;;
    esac
fi
echo $$ > "$state/pid"
# Prefer not to be the one killed under pressure (best effort).
{ echo -1000 > /proc/self/oom_score_adj; } 2>/dev/null

page=$(getconf PAGESIZE 2>/dev/null)
case "$page" in ''|*[!0-9]*) page=4096 ;; esac

cache_value() {  # $1 cache key, $2 shmem key, $3 file -> $value = cache - shmem (0 when absent)
    cache=0
    shmem=0
    value=0
    [ -r "$3" ] || return 0
    while read -r k v; do
        case "$k" in
            "$1") cache=$v ;;
            "$2") shmem=$v ;;
        esac
    done < "$3"
    case "$cache$shmem" in *[!0-9]*) cache=0; shmem=0 ;; esac
    value=$((cache - shmem))
    [ "$value" -ge 0 ] || value=0
}

read_used() {  # -> $used (bytes); non-zero status when unreadable
    used=
    case "$src" in
        cgroup2)
            read -r used < /sys/fs/cgroup/memory.current || [ -n "$used" ] || return 1
            value=0
            cache_value file shmem /sys/fs/cgroup/memory.stat 2>/dev/null
            used=$((used - value)) ;;
        cgroup1)
            read -r used < /sys/fs/cgroup/memory/memory.usage_in_bytes || [ -n "$used" ] || return 1
            value=0
            cache_value total_cache total_shmem /sys/fs/cgroup/memory/memory.stat 2>/dev/null
            used=$((used - value)) ;;
        meminfo)
            total=
            avail=
            while read -r k v _; do
                case "$k" in
                    MemTotal:) total=$v ;;
                    MemAvailable:) avail=$v; break ;;
                esac
            done < /proc/meminfo
            [ -n "$total" ] && [ -n "$avail" ] || return 1
            used=$(( (total - avail) * 1024 )) ;;
        file:*)
            read -r used < "${src#file:}" || [ -n "$used" ] || return 1 ;;
        *) return 1 ;;
    esac
    case "$used" in ''|*[!0-9-]*) used=; return 1 ;; esac
    return 0
} 2>/dev/null

if [ "$src" = auto ]; then
    src=none
    if [ -r /sys/fs/cgroup/memory.current ]; then
        src=cgroup2
    elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        src=cgroup1
    else
        case "$(uname -r 2>/dev/null)" in *gvisor*) src=meminfo ;; esac
    fi
fi
if [ "$src" = none ] || ! read_used; then
    echo "source=none" > "$state/status"
    rm -f "$state/pid"
    exit 0
fi
echo "source=$src used=$used page=$page" > "$state/status"

seq=0
if [ -r "$state/events.tsv" ]; then
    while IFS='	' read -r s _; do
        case "$s" in ''|*[!0-9]*) ;; *) seq=$s ;; esac
    done < "$state/events.tsv"
fi

event() {  # kind pid rss command
    seq=$((seq + 1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$seq" "$(date +%s)" "$1" \
        "$used" "$limit" "$2" "$3" "$4" >> "$state/events.tsv"
}

find_victim() {  # -> $vpid, $vrss (bytes); vpid empty when none
    vpid=
    vrss=0
    for d in /proc/[0-9]*; do
        pid=${d#/proc/}
        [ "$pid" = 1 ] && continue
        [ "$pid" = "$$" ] && continue
        rss=
        read -r _ rss _ < "$d/statm" 2>/dev/null || continue
        case "$rss" in ''|*[!0-9]*) continue ;; esac
        [ "$rss" -gt "$vrss" ] || continue
        if [ -n "$scope" ]; then
            line=
            read -r line < "$d/stat" 2>/dev/null || continue
            rest=${line##*) }
            # shellcheck disable=SC2086
            set -- $rest
            [ "${4:-}" = "$scope" ] || continue
        fi
        adj=0
        read -r adj < "$d/oom_score_adj" 2>/dev/null
        [ "$adj" = -1000 ] && continue
        vpid=$pid
        vrss=$rss
    done
    vrss=$((vrss * page))
}

# A process smaller than this is not worth killing: the memory is elsewhere
# (tmpfs, many small processes) and killing frees little.
min_victim=$((limit / 50))
last_nokill=0
tick=0
while [ "$tick" -lt "$ticks" ]; do
    tick=$((tick + 1))
    sleep 1
    read_used || continue
    [ "$used" -gt "$threshold" ] || continue
    find_victim
    if [ -z "$vpid" ] || [ "$vrss" -lt "$min_victim" ]; then
        if [ $((tick - last_nokill)) -ge 60 ] || [ "$last_nokill" = 0 ]; then
            last_nokill=$tick
            event nokill "${vpid:-0}" "$vrss" ""
        fi
        continue
    fi
    cmd=$(tr '\0\t\n' '   ' < "/proc/$vpid/cmdline" 2>/dev/null | cut -c1-200)
    if [ -z "$cmd" ]; then
        line=
        read -r line < "/proc/$vpid/stat" 2>/dev/null
        cmd=${line#*(}
        cmd="[${cmd%%)*}]"
    fi
    if kill -KILL "$vpid" 2>/dev/null; then
        event kill "$vpid" "$vrss" "$cmd"
        # Let the kernel release the memory before measuring again.
        sleep 2
        tick=$((tick + 2))
    fi
done
event expire 0 0 ""
rm -f "$state/pid"
exit 0
