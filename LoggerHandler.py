# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:43:57 2023

@author: Magnus
"""

import logging

class LoggerHandler:
    
    @staticmethod
    def create_PlayConnect4Logger() -> logging.Logger:
        logger = logging.getLogger("PlayConnect4")
        logger.setLevel(logging.DEBUG)
        
        if logger.hasHandlers():
            return logger
        
        # Create handlers
        file_handler = logging.FileHandler('logs/PlayConnect4.log')
        file_handler.setLevel(logging.DEBUG)
        
        #Create formatters and add it to handlers
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        
        return logger
    
    
if __name__ == "__main__":
    logger = LoggerHandler.create_PlayConnect4Logger()
    logger.info("info test")
    logger.debug("debug test")
