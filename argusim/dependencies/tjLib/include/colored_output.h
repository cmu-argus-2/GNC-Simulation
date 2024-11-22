/****************************************************************
 *
 * @brief 		Header file specifying macros to use in order to create
 *a string of a certain color on output console
 *
 * @authors 	Ben Kolligs, ...
 * @author 		Carnegie Mellon University, Planetary Robotics Lab
 *
 ****************************************************************/
#ifndef _colored_output_h_
#define _colored_output_h_

#define ESCAPE    "\033["
#define TERMINATE "m"
/* Reset color back to normal with ANSI color codes */
#define RESET_COLOR "0" TERMINATE
#define END_COLOR   ESCAPE RESET_COLOR

/* Define ANSI color codes for strings */
#define COLOR_RED      "31" TERMINATE
#define COLOR_RED_BOLD "1;" COLOR_RED
#define COLOR_GREEN    "32" TERMINATE
#define GREEN_BOLD     "1;" COLOR_GREEN
#define COLOR_YELLOW   "33" TERMINATE
#define YELLOW_BOLD    "1;" COLOR_YELLOW
#define COLOR_BLUE     "34" TERMINATE
#define BLUE_BOLD      "1;" COLOR_BLUE
#define COLOR_MAGENTA  "35" TERMINATE
#define MAGENTA_BOLD   "1;" COLOR_MAGENTA
#define COLOR_CYAN     "36" TERMINATE
#define CYAN_BOLD      "1;" COLOR_CYAN

/* Define macros to process different color strings */
#define RED_STRING(string)          ESCAPE COLOR_RED string END_COLOR
#define RED_BOLD_STRING(string)     ESCAPE COLOR_RED_BOLD string END_COLOR
#define GREEN_STRING(string)        ESCAPE COLOR_GREEN string END_COLOR
#define GREEN_BOLD_STRING(string)   ESCAPE GREEN_BOLD string END_COLOR
#define YELLOW_STRING(string)       ESCAPE COLOR_YELLOW string END_COLOR
#define YELLOW_BOLD_STRING(string)  ESCAPE YELLOW_BOLD string END_COLOR
#define BLUE_STRING(string)         ESCAPE COLOR_BLUE string END_COLOR
#define BLUE_BOLD_STRING(string)    ESCAPE BLUE_BOLD string END_COLOR
#define MAGENTA_STRING(string)      ESCAPE COLOR_MAGENTA string END_COLOR
#define MAGENTA_BOLD_STRING(string) ESCAPE MAGENTA_BOLD string END_COLOR
#define CYAN_STRING(string)         ESCAPE COLOR_CYAN string END_COLOR
#define CYAN_BOLD_STRING(string)    ESCAPE CYAN_BOLD string END_COLOR

/* Define macros to represent the type of information we are presenting */
#define ERROR_INFO(string)   RED_STRING(string)
#define WARNING_INFO(string) YELLOW_STRING(string)
#define SUCCESS_INFO(string) GREEN_STRING(string)

#endif   //_colored_output_h_ header