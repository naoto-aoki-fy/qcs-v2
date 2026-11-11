#pragma once

#include <iomanip>
#include <ostream>

inline void writeTimingCsvHeader(std::ostream& output) {
    output << "type,sample,repetitions,qubit,milliseconds\n";
}

inline void writeTimingCsvRow(std::ostream& output, unsigned long long sample,
                              unsigned int repetitions, double milliseconds,
                              int qubit = -1) {
    output << (qubit < 0 ? "total" : "gate") << ',' << sample << ','
           << repetitions << ',';
    if (qubit >= 0) output << qubit;
    output << ',' << std::fixed << std::setprecision(6) << milliseconds << '\n';
}
