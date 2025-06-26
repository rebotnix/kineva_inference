import kineva.industrial.modbus.rb_modbus as rb_m
import time

s = rb_m.MODBUS(host="127.0.0.1",port=1502)
s.start()

while True:
  print(rb_m.rb_words)
  time.sleep(1)
