#!/bin/bash
 
TARGET_MAC="30:23:00:00:20:81"
LOGFILE="/home/mdn/sens-able-prama/speech-assistant/bluetooth_connect.log"
 
echo "$(date): Unblocking Bluetooth if blocked..." | tee -a "$LOGFILE"
rfkill unblock bluetooth
 
#echo "$(date): Bringing up hci0 interface..." | tee -a "$LOGFILE"
#hciconfig hci0 up
 
echo "$(date): Attempting to power on Bluetooth..." | tee -a "$LOGFILE"
POWER_OUTPUT=$(bluetoothctl power on 2>&1)
if echo "$POWER_OUTPUT" | grep -qi "Failed"; then
    echo "$(date): Failed to power on Bluetooth: $POWER_OUTPUT" | tee -a "$LOGFILE"
    exit 1
else
    echo "$(date): Bluetooth powered on successfully." | tee -a "$LOGFILE"
fi
 
echo "$(date): Starting Bluetooth scan..." | tee -a "$LOGFILE"
bluetoothctl scan on &
SCAN_PID=$!
sleep 7
 
if bluetoothctl devices | grep "$TARGET_MAC"; then
    echo "$(date): Found target $TARGET_MAC. Pairing and connecting..." | tee -a "$LOGFILE"
    bluetoothctl pair $TARGET_MAC
    sleep 2
    bluetoothctl trust $TARGET_MAC
    sleep 2
    bluetoothctl connect $TARGET_MAC
    sleep 2
    echo "$(date): Connection attempt done." | tee -a "$LOGFILE"
else
    echo "$(date): Device $TARGET_MAC not found." | tee -a "$LOGFILE"
fi
 
kill $SCAN_PID 2>/dev/null
