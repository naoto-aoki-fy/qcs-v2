#include "../src/qubit_order.hpp"
#include <cassert>
#include <iostream>

int main() {
    assert(parseQubitOrder("ascending") == QubitOrder::Ascending);
    assert(parseQubitOrder("descending") == QubitOrder::Descending);
    for (const auto* value : {"", "reverse", "Ascending"}) {
        bool rejected = false;
        try { parseQubitOrder(value); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    // The final all-Hadamard state is identical in either order; check targets directly.
    for (unsigned n = 1; n <= 59; ++n) {
        for (const auto order : {QubitOrder::Ascending, QubitOrder::Descending}) {
            bool seen[59] = {};
            for (unsigned gate = 0; gate < n; ++gate) {
                const unsigned q = targetQubit(gate, n, order);
                assert(q < n && !seen[q]);
                seen[q] = true;
                if (gate == 0) assert(q == (order == QubitOrder::Ascending ? 0 : n - 1));
                else assert(q == (order == QubitOrder::Ascending ? gate : n - 1 - gate));
            }
        }
    }
    std::cout << "Qubit order tests passed\n";
}
