from kineva.industrial.modbus.core.client import ModbusClient
import time
import random
from datetime import datetime

counter = 0

def get_current_time():
    # Get the current time
    now = datetime.now()
    # Extract the year, month, day, hour, minute, second, and millisecond
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    millisecond = now.microsecond // 1000  # Convert microseconds to milliseconds
    # Return the values as integers
    return year, month, day, hour, minute, second, millisecond

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1502

# Initialize Modbus client
client = ModbusClient(host=SERVER_HOST, port=SERVER_PORT)

def send_modbus_message(trigger):
    global counter
    """Send an array of integers to the Modbus server."""
    try:

        # Ensure the client is connected
        #if not client.is_open():
        #    client.open()

        year, month, day, hour, minute, second, millisecond = get_current_time()
        data = [year, month, day, hour, minute, second, millisecond, trigger, 0, 0, counter, counter, counter, counter, counter, counter, counter, counter,0,0,0,0,0,0,0]

        # Write the array to holding registers starting at address 0
        success = client.write_multiple_registers(0, data)

        #counter = counter + 1

    except Exception as e:
        print(f"Error: {e}")

def run_simulation():
    ctr_c = 0
    trigger = 0
    """Send messages to the server every second."""
    while True:
        if ctr_c > 3:
          ctr_c = 0
          if trigger == 0:
            trigger = 1
          elif trigger == 1:
            trigger = 2
          elif trigger == 2:
            trigger = 3
          else:
            trigger = 0

        send_modbus_message(trigger)
        ctr_c = ctr_c + 1
        time.sleep(1)  # Wait for 1 second

if __name__ == "__main__":
    run_simulation()
