/* Copyright (c) 1993-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <fstream>
#include <iterator>
#include "ioHelper.h"
using namespace std;

namespace nvinfer1
{
	
string getBasename(string const& path)
{
#ifdef _WIN32
    constexpr char SEPARATOR = '\\';
#else
    constexpr char SEPARATOR = '/';
#endif
    int baseId = path.rfind(SEPARATOR) + 1;
    return path.substr(baseId, path.rfind('.') - baseId);
}

ostream& operator<<(ostream& o, const nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
    case ILogger::Severity::kINTERNAL_ERROR: o << "INTERNAL_ERROR"; break;
    case ILogger::Severity::kERROR: o << "ERROR"; break;
    case ILogger::Severity::kWARNING: o << "WARNING"; break;
    case ILogger::Severity::kINFO: o << "INFO"; break;
    }
    return o;
}

void writeBuffer(void* buffer, size_t size, string const& path)
{
    ofstream stream(path.c_str(), ios::binary);

    if (stream)
        stream.write(static_cast<char*>(buffer), size);
}

// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
    string buffer;
    ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

} // namespace nvinfer1
