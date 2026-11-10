#pragma once

#include <stdexcept>
#include <string>

enum class QubitOrder { Ascending, Descending };

inline QubitOrder parseQubitOrder(const std::string& value) {
    if (value == "ascending") return QubitOrder::Ascending;
    if (value == "descending") return QubitOrder::Descending;
    throw std::invalid_argument("--qubit-order requires ascending or descending");
}

// gate is the execution position, in [0, qubits).
inline unsigned int targetQubit(unsigned int gate, unsigned int qubits, QubitOrder order) {
    return order == QubitOrder::Descending ? qubits - 1 - gate : gate;
}
