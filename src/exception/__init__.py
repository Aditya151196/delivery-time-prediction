import os,sys

class CustomException(Exception):
    def __init__(self,error_message:Exception,error_detail:sys):
        self.error_message=CustomException.get_detailed_error_message(error_message=error_message,
                                                                      error_detail=error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_detail:sys)->str:
        _,_,exec_tb = error_detail.exc_info()

        exception_block_line_number=exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message=f"""
        Error occured in execution of :
        [{file_name}] 
        at try_block_line_number : [{try_block_line_number}] 
        and exception_block_line_number : [{exception_block_line_number}]
        error_message : {error_message}]
        """

        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return CustomException.__name__.str()
        



        