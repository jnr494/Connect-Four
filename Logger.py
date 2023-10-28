# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 12:43:57 2023

@author: Magnus
"""

import logging

class Logger:
    
    @staticmethod
    def create_PlayConnect4Logger() -> logging.Logger:
        logger = logging.getLogger("PlayConnect4")
        logger.setLevel(logging.DEBUG)
        
        if logger.hasHandlers():
            return logger
        
        # Create handlers
        file_handler = logging.FileHandler('logs/PlayConnect4.log')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        #Create formatters and add it to handlers
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        print(logger.handlers)
        return logger
    
    
if __name__ == "__main__":
    logger = Logger.create_PlayConnect4Logger()
    logger.info("info test")
    logger.debug("debug test")
