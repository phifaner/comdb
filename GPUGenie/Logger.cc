/*! \file Logger.cc
 *  \brief Implementation for Logger.h
 */

#include "Logger.h"
#include <stdarg.h>
#include <string.h>
#include <sstream>
#include <sys/time.h>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

#include "Timing.h"
using namespace std;
const char * const Logger::LEVEL_NAMES[] =
{ "NONE   ", "ALERT  ", "INFO   ", "VERBOSE", "DEBUG  " };

Logger * Logger::logger = NULL;

Logger::Logger(int level)
{
	log_level = level;
	std::string s = currentDateTime();
	char fout_name[128];
	sprintf(fout_name, "GPUGENIE_LOG-%s.log", s.c_str());
    stringstream ss;
    ss<<"log/"<<string(fout_name);
    struct stat st;
    if(stat("log", &st) == -1)
        mkdir("log", 0700);
	strcpy(logfile_name, ss.str().c_str());
	logfile = fopen(logfile_name, "a");
}

Logger::~Logger()
{
	fclose(logfile);
}

void Logger::exit(void)
{
	if (logger != NULL)
	{
		log(VERBOSE, "---------Exiting  Logger----------");
		delete logger;
	}

}

Logger* Logger::_logger(void)
{
	if (logger == NULL)
	{
		logger = new Logger(INFO);
		log(VERBOSE, "---------Starting Logger %s----------",
				logger->logfile_name);

	}
	return logger;
}

void Logger::set_level(int level)
{
	_logger()->log_level = level;
}
int Logger::get_level()
{
	return _logger()->log_level;
}

void Logger::set_logfile_name(const char * name)
{
	if (strcmp(name, _logger()->logfile_name) != 0)
	{
		strcpy(_logger()->logfile_name, name);
		if (logger != NULL)
		{
			fclose(logger->logfile);
			logger = NULL;
			_logger();
		}

	}

}

const char * Logger::get_logfile_name()
{
	return _logger()->logfile_name;
}

int Logger::log(int level, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);

	timeval curTime;
	gettimeofday(&curTime, NULL);
	int milli = curTime.tv_usec / 1000;

	char buffer[80];
	strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

	char currentTime[84];
	sprintf(currentTime, "[%s:%03d %s] ", buffer, milli, LEVEL_NAMES[level]);
	fprintf(_logger()->logger->logfile, currentTime);

	char message[1024];
	vsprintf(message, fmt, args);
	va_end(args);

	fprintf(_logger()->logger->logfile, message);
	fprintf(_logger()->logger->logfile, "\n");

	if (_logger()->logger->log_level >= level)
	{
		printf(message);
		printf("\n");
		return 1;
	}
	else
	{
		return 0;
	}
}
