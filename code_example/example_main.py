#
# example_main.py
#

import optparse
import sys

class A:
    def __init__(self):
        print("This program is boring!")
        
        
#
# Main entry point
#
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        (options, args) = parser.parse_args(argv)
                
        #do something here...
        #
        a=A()
                    
        if not options.quietMode:                    
            print('Main Completed!')    
    
    except Exception as info:
        if 'options' in vars() and not options.quietMode:
            raise   #re-raise Exception, interpreter shows stack trace
        else:
            print(info)   
    

if __name__ == '__main__':
    main()
    