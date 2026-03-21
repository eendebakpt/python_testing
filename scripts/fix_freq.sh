# Frequencies are in KHz. Set min == max to lock it.
# E.g., to lock at 2.0 GHz:


echo "Disable hw managed pstate"
echo passive | sudo tee /sys/devices/system/cpu/intel_pstate/status
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo


echo "Fix cpu frequency"
echo 3000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
echo 3000000 | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq

echo "disable deep-C states"
for cpu in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]; do
    #echo 1 | sudo tee "$cpu/disable" 2>/dev/null
    echo 1 | sudo "$cpu/disable" 2>/dev/null
done



echo "scaling_cur_freq for cpu 0"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq



