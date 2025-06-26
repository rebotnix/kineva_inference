from .core.server import ModbusServer, DataHandler, DataBank
from .core.constants import EXP_ILLEGAL_FUNCTION
import logging
from threading import Thread
from .core.constants import READ_COILS, READ_DISCRETE_INPUTS, READ_HOLDING_REGISTERS, READ_INPUT_REGISTERS, \
    WRITE_MULTIPLE_COILS, WRITE_MULTIPLE_REGISTERS, WRITE_SINGLE_COIL, WRITE_SINGLE_REGISTER, \
    EXP_NONE, EXP_ILLEGAL_FUNCTION, EXP_DATA_ADDRESS, EXP_DATA_VALUE

# some const
ALLOW_R_L = ['127.0.0.1', '192.168.0.4', '10.24.87.20', '10.24.87.15']
ALLOW_W_L = ['127.0.0.1', '192.168.0.4', '10.24.87.20', '10.24.87.15']

initial_run = True
rb_lock = False
rb_words = None
rb_write_words = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
read_c = 1

class mb_databank(DataBank):
    """A custom ModbusServerDataBank for override on_xxx_change methods."""

    #def __init__(self):
      #  self.i = 0

    def on_coils_change(self, address, from_value, to_value, srv_info):
        """Call by server when change occur on coils space."""
        msg = 'change in coil space [{0!r:^5} > {1!r:^5}] at @ 0x{2:04X} from ip: {3:<15}'
        msg = msg.format(from_value, to_value, address, srv_info.client.address)
        #print(msg)
        logging.info(msg)

    def on_holding_registers_change(self, address, from_value, to_value, srv_info):
        """Call by server when change occur on holding registers space."""
        #self.i = self.i+1
       
        #msg = 'change in hreg space [{0!r:^5} > {1!r:^5}] at @ 0x{2:04X} from ip: {3:<15}'
        msg = 'change in hreg space [{0!r:^5} > {1!r:^5}] at @ 0x{2:04X} from ip: {3:<15}'
        msg = msg.format(from_value, to_value, address, srv_info.client.address)
        msg2 = msg.format(from_value, to_value, address, srv_info.client.address)
        #print(from_value, to_value,i)
        #print(from_value,to_value,address)
        #print("VALUE:" + str(to_value), "ADDR:" + str(address), srv_info.client.address)

    def rebotnix_callback(self,bytes_from_gateway):
        print(bytes_from_gateway)

    #def get_holding_registers(address, number=32, srv_info=None):
    #    print(address,number)

        #print("-")
        #logging.info(msg)

# a custom data handler with IPs filter
class mb_datahandler(DataHandler):

    def read_coils(self, address, count, srv_info):
        print("1")
        if srv_info.client.address in ALLOW_R_L:
            return super().read_coils(address, count, srv_info)
        else:
            return DataHandler.Return(exp_code=EXP_ILLEGAL_FUNCTION)

    def read_d_inputs(self, address, count, srv_info):
        print("2")
        if srv_info.client.address in ALLOW_R_L:
            return super().read_d_inputs(address, count, srv_info)
        else:
            return DataHandler.Return(exp_code=EXP_ILLEGAL_FUNCTION)

    def write_coils(address_bits_1,srv_info):
        print("3")

    def on_holding_registers_change(self, address, from_value, to_value, srv_info):
        print(address)

    def read_i_regs(self, address, count, srv_info):
        """Call by server for reading in the input registers space
        :param address: start address
        :type address: int
        :param count: number of input registers
        :type count: int
        :param srv_info: some server info
        :type srv_info: ModbusServer.ServerInfo
        :rtype: Return
        """
        #print("RB read_i_regs.")
        # read words from DataBank
        words_l = self.data_bank.get_input_registers(address, count, srv_info)
        #print("WORDS read_i_regs: "+str(words_l))
        # return DataStatus to server
        if words_l is not None:
            return DataHandler.Return(exp_code=EXP_NONE, data=words_l)
        else:
            return DataHandler.Return(exp_code=EXP_DATA_ADDRESS)

    def read_h_regs(self, address, count, srv_info):
        """Call by server for reading in the holding registers space
        :param address: start address
        :type address: int
        :param count: number of holding registers
        :type count: int
        :param srv_info: some server info
        :type srv_info: ModbusServer.ServerInfo
        :rtype: Return
        """
        #print("RB read_h_regs.")

        # read words from DataBank
        words_l = self.data_bank.get_holding_registers(address, count, srv_info)
        #print("WORDS read_h_regs: "+str(words_l))

        # return DataStatus to server
        if words_l is not None:
            return DataHandler.Return(exp_code=EXP_NONE, data=words_l)
        else:
            return DataHandler.Return(exp_code=EXP_DATA_ADDRESS)


    # we recieve this register from modbus
    def write_h_regs(self, address, words_l, srv_info):
        global rb_lock
        global rb_words
        global initial_run
        global read_c
        global rb_write_words

        """Call by server for reading in the holding registers space

        :param address: start address
        :type address: int
        :param count: number of holding registers
        :type count: int
        :param srv_info: some server info
        :type srv_info: ModbusServer.ServerInfo
        :rtype: Return
        """
        # read words from DataBank
        #print("Words: "+str(words_l))

        if not initial_run:
            print("RB MODBUS IS RECEIVING DATA.")
            rb_lock = True
            rb_words = words_l
            #print(rb_words)

            while rb_lock:
                x = 1

        initial_run = False

        address = 32
        count = 16
        if rb_words != None:
          self.data_bank.set_input_registers(address, rb_write_words)
        #read_c = read_c + 1
        update_ok = self.data_bank.set_holding_registers(32, rb_write_words, srv_info)
        #self.data_bank.set_input_registers(40, [2,4,6,8,10,12,14,16])

        # return DataStatus to server
        if update_ok:
          #self.read_h_regs(33, count, srv_info)
          #self.read_i_regs(41, count, srv_info)

          return DataHandler.Return(exp_code=EXP_NONE)
        else:
          return DataHandler.Return(exp_code=EXP_DATA_ADDRESS)


class MODBUS(Thread):
    def __init__(self, host="192.168.0.7", port=1502):
        Thread.__init__(self)
        self.host = host
        self.port = port

        self.last_item = None
        self.datahandler = mb_datahandler(mb_databank())

    def run(self):
        #start modbus server
        self.server = ModbusServer(host=self.host, port=self.port, no_block=False, data_hdl=self.datahandler)
        self.server.start()

    def stop(self):
        self.server.stop()
