# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:19:31 2018

@author: Python
"""

import logging
import sys

class LogHelper:
    
    def __init__(self, name='LogHelper', 
                 setLevel=logging.DEBUG):
       self.logger = logging.getLogger(name)
       self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
       
       self.file_handler = logging.FileHandler(name+".log")
       self.file_handler.setFormatter(self.formatter)
       
       self.consle_handler = logging.StreamHandler(sys.stdout)
       self.consle_handler.setFormatter(self.formatter)
       
       self.logger.setLevel(setLevel)
       
       self.logger.addHandler(self.file_handler)
       self.logger.addHandler(self.consle_handler)
    
    def writeLog(self, info, level='debug'):
        if level == "critial":
            self.logger.critical(info)
        elif level == "error":
            self.logger.error(info)
        elif level == "warning":
            self.logger.warning(info)
        elif level == "info":
            self.logger.info(info)
        else:
            self.logger.debug(info)
            
    def removeLog(self):
        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.consle_handler)
        
if __name__ == "__main__":
    logger = LogHelper()
    logger.writeLog("helloworld", level='error')
    logger.removeLog()
       
       
       
       
       
       
       
       
       