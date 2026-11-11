#include "../src/timing_csv.hpp"

#include <iostream>
#include <sstream>

int main() {
    std::ostringstream output;
    writeTimingCsvHeader(output);
    writeTimingCsvRow(output, 1, 2, 0.125);
    writeTimingCsvRow(output, 1, 2, 0.0, 7);
    writeTimingCsvRow(output, 2, 2, 0.000001, 0);
    const std::string expected =
        "type,sample,repetitions,qubit,milliseconds\n"
        "total,1,2,,0.125000\n"
        "gate,1,2,7,0.000000\n"
        "gate,2,2,0,0.000001\n";
    if (output.str() != expected) {
        std::cerr << "Unexpected timing CSV: " << output.str();
        return 1;
    }
}
