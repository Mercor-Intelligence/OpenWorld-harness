# Memory sample for the host-side poller (memwatch.py parse_sample).
#
# Sourced after the poll's preamble, which sets:
#   state       the watchdog's state directory (memwatch.sh)
#   top_n       how many of the largest processes to list
#   top_groups  how many of the largest process groups to list
#   gpu         1 to read nvidia-smi as well
#
# Prints one tagged, tab-separated line per reading, a few lines whatever the
# number of processes:
#   mem    cg=<1|2|> usage= max= epoch= src=<loop source> sused=<its reading,
#          for the non-cgroup sources; parse_sample computes the cgroup ones>
#   stat   <key>=<bytes>       selected memory.stat keys
#   peak   <bytes>             the loop's highest reading (memwatch.sh)
#   procs  <rss pages> <n>     every process with memory
#   grp    <pgid> <rss pages> <n>
#   top    <pid> <pgid> <rss pages> <comm> <command line, 100 chars>
#   df     <mount> <used KiB> <size KiB>
#   gpu    <used MiB> <total MiB>
#
# POSIX sh: builtins plus sort, head, tr, cut, df and nvidia-smi when present.
# shellcheck disable=SC2154  # state, top_n, top_groups, gpu: set by the preamble

cg=
usage=
maxu=
stat=
if [ -r /sys/fs/cgroup/memory.current ]; then
    cg=2
    read -r usage < /sys/fs/cgroup/memory.current
    read -r maxu < /sys/fs/cgroup/memory.peak 2>/dev/null
    stat=/sys/fs/cgroup/memory.stat
elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
    cg=1
    read -r usage < /sys/fs/cgroup/memory/memory.usage_in_bytes
    read -r maxu < /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null
    stat=/sys/fs/cgroup/memory/memory.stat
fi

# The loop's own source and reading (as memwatch.sh read_used).
src=
read -r src _ < "$state/status" 2>/dev/null
src=${src#source=}
sused=
case "$src" in
    meminfo)
        total=
        avail=
        while read -r k v _; do
            case "$k" in
                MemTotal:) total=$v ;;
                MemAvailable:) avail=$v; break ;;
            esac
        done < /proc/meminfo
        [ -n "$total" ] && [ -n "$avail" ] && sused=$(( (total - avail) * 1024 )) ;;
    file:*)
        read -r sused < "${src#file:}" 2>/dev/null ;;
esac
printf 'mem\tcg=%s usage=%s max=%s epoch=%s src=%s sused=%s\n' \
    "$cg" "$usage" "$maxu" "$(date +%s)" "${src%%:*}" "$sused"

if [ -n "$stat" ] && [ -r "$stat" ]; then
    while read -r k v; do
        case "$k" in
            total_cache|total_shmem|total_rss|total_mapped_file|total_inactive_file|\
            cache|shmem|rss|anon|file|inactive_file|file_mapped)
                printf 'stat\t%s=%s\n' "$k" "$v" ;;
        esac
    done < "$stat"
fi

peak=
read -r peak < "$state/peak" 2>/dev/null
printf 'peak\t%s\n' "$peak"

# pid pgid rss-pages comm, for every process with memory.
procs=$(for d in /proc/[0-9]*; do
    rss=
    read -r _ rss _ < "$d/statm" 2>/dev/null || continue
    case "$rss" in ''|0|*[!0-9]*) continue ;; esac
    line=
    read -r line < "$d/stat" 2>/dev/null || continue
    comm=${line#*(}
    comm=${comm%)*}
    rest=${line##*) }
    # shellcheck disable=SC2086
    set -- $rest
    printf '%s %s %s %s\n' "${d#/proc/}" "${3:-0}" "$rss" "$comm"
done)

# Per-group sums (G pgid pages n) and the total (T pages n).
agg=$(printf '%s\n' "$procs" | sort -k2,2n 2>/dev/null | {
    g=
    gs=0
    gn=0
    ts=0
    tn=0
    while read -r pid pgid rss _; do
        [ -n "$pid" ] || continue
        ts=$((ts + rss))
        tn=$((tn + 1))
        if [ "$pgid" != "$g" ]; then
            [ -n "$g" ] && echo "G $g $gs $gn"
            g=$pgid
            gs=0
            gn=0
        fi
        gs=$((gs + rss))
        gn=$((gn + 1))
    done
    [ -n "$g" ] && echo "G $g $gs $gn"
    echo "T $ts $tn"
})
printf '%s\n' "$agg" | while read -r t a b _; do
    [ "$t" = T ] && printf 'procs\t%s\t%s\n' "$a" "$b"
done
printf '%s\n' "$agg" | while read -r t a b c; do
    [ "$t" = G ] && echo "$a $b $c"
done | sort -k2,2nr 2>/dev/null | head -n "$top_groups" | while read -r g gs gn; do
    printf 'grp\t%s\t%s\t%s\n' "$g" "$gs" "$gn"
done

printf '%s\n' "$procs" | sort -k3,3nr 2>/dev/null | head -n "$top_n" |
while read -r pid pgid rss comm; do
    [ -n "$pid" ] || continue
    cmd=$(tr '\0\t\n' '   ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)
    printf 'top\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$pgid" "$rss" "$comm" "$cmd"
done

df -kP /dev/shm /tmp 2>/dev/null | while read -r _ blocks used _ _ mnt; do
    case "$used" in ''|*[!0-9]*) continue ;; esac
    printf 'df\t%s\t%s\t%s\n' "$mnt" "$used" "$blocks"
done

if [ "$gpu" = 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
    timeout 5 nvidia-smi --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -n 8 |
    while IFS=', ' read -r u t; do
        printf 'gpu\t%s\t%s\n' "$u" "$t"
    done
fi
true
