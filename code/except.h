/*++

Copyright (c) 2024  Dr. Chang Liu, PhD.

Module Name:

    except.h

Abstract:

    This header file contains definitions for all the exception classes thrown by
    this program.

Revision History:

    2024-11-13  File created

--*/

#pragma once

#include <string>
#include <sstream>
#include <exception>

#define ThrowException(Ex, ...)					\
    throw Ex(__FILE__, __LINE__, __func__, ##__VA_ARGS__)

// Base exception class that records the file name, line number, and function name
// where the exception has happened.
class Exception : public std::exception {
protected:
    std::string msg;
public:
    Exception(const char *file, int line, const char *func) {
	std::stringstream ss;
	ss << file << ":" << line << "@" << func << "(): ";
	msg = ss.str();
    }

    virtual const char *what() const noexcept {
	return msg.c_str();
    }
};

// Exception class thrown when the user passed in an invalid argument to a function
class InvalidArgument : public Exception {
public:
    InvalidArgument(const char *f, int l, const char *fn,
		    const char *name, const char *param_msg) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Invalid argument for " << name << ": "
	   << name << " " << param_msg << ".";
	msg += ss.str();
    }

    InvalidArgument(const char *f, int l, const char *fn,
		    const char *err_msg) : Exception(f, l, fn) {
	msg += std::string("Invalid argument: ") + err_msg;
    }
};

// Exception class thrown when a runtume error has occured
class RuntimeError : public Exception {
public:
    RuntimeError(const char *f, int l, const char *fn,
		 const char *lib, int code) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << lib << " runtime error, code " << code << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a libmatio failed to open a file
class MatioCreateFileError : public Exception {
public:
    MatioCreateFileError(const char *f, int l, const char *fn,
			 const char *var) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "libmatio: failed to open/create file " << var << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a libmatio failed to create a variable
class MatioCreateVarError : public Exception {
public:
    MatioCreateVarError(const char *f, int l, const char *fn,
			const char *var) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "libmatio: failed to create variable " << var << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a libmatio failed to create a variable
class MatioReadVarError : public Exception {
public:
    MatioReadVarError(const char *f, int l, const char *fn,
			const char *var) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "libmatio: failed to read variable " << var << ".";
	msg += ss.str();
    }
};

// Exception class thrown when a libmatio failed to create a variable
class MatioInvalidVarError : public Exception {
public:
    MatioInvalidVarError(const char *f, int l, const char *fn,
			 const char *var, const char *reason) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << "libmatio: variable " << var << " is invalid (" << reason << ")";
	msg += ss.str();
    }
};

#ifndef NOGPU
// Exception class thrown when a runtume error has occured
class CudaError : public Exception {
public:
    CudaError(const char *f, int l, const char *fn, cudaError_t code) : Exception(f, l, fn) {
	std::stringstream ss;
	ss << " GPU runtime error, code " << code << " ("
	   << cudaGetErrorString(code) << ").";
	msg += ss.str();
    }
};
#endif

// Exception class thrown when there is no GPU device available
class NoGpuDevice : public Exception {
public:
    NoGpuDevice(const char *f, int l, const char *fn)
	: Exception(f, l, fn) {
	msg += "No GPU device available";
    }
};

class NoP2PAccess : public Exception {
public:
    NoP2PAccess(const char *f, int l, const char *fn, int dev0, int dev1)
	: Exception(f, l, fn) {
	std::stringstream ss;
	ss << "P2P access between device " << dev0 << " and device " << dev1
	   << " is not supported.";
	msg += ss.str();
    }
};

// Exception class thrown when a GPU device memory allocation has failed
class DevOutOfMem : public Exception {
public:
    DevOutOfMem(const char *f, int l, const char *fn, size_t size)
	: Exception(f, l, fn) {
	std::stringstream ss;
	ss << "Failed to allocate " << size << " bytes on GPU device.";
	msg += ss.str();
    }
};
