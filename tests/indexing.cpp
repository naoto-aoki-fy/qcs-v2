#include "../src/indexing.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    for (unsigned n = 1; n <= 12; ++n) {
        const Index pairs = Index{1} << (n - 1);
        for (unsigned q = 0; q < n; ++q) {
            std::vector<unsigned> visits(2 * pairs);
            for (Index p = 0; p < pairs; ++p) {
                const Index a = firstAmplitude(p, Index{1} << q);
                assert(a < 2 * pairs && !(a & (Index{1} << q)));
                ++visits[a];
                ++visits[a | (Index{1} << q)];
            }
            for (auto count : visits) assert(count == 1);
        }
        for (unsigned block : {64u, 128u, 256u}) {
            for (unsigned limit : {1u, 3u, 2147483647u}) {
                const unsigned grid = gridFor(pairs, block, limit);
                assert(grid <= limit);
                std::vector<unsigned> visits(pairs);
                const Index stride = static_cast<Index>(grid) * block;
                for (Index thread = 0; thread < stride; ++thread)
                    for (Index p = thread; p < pairs; p += stride) ++visits[p];
                for (auto count : visits) assert(count == 1);
            }
        }
    }
    for (unsigned q = 0; q < 59; ++q) {
        for (Index p : {Index{0}, (Index{1} << 31) - 1, Index{1} << 32,
                        (Index{1} << 58) - 1}) {
            const Index bit = Index{1} << q;
            const Index a = firstAmplitude(p, bit);
            const Index reference = ((p >> q) << (q + 1)) | (p & (bit - 1));
            assert(a == reference);
            assert(a < (Index{1} << 59));
            assert(((a & (bit - 1)) | ((a >> (q + 1)) << q)) == p);
        }
    }
    assert(gridFor(Index{1} << 58, 64, 2147483647u) == 2147483647u);
    assert(gridFor((Index{1} << 32) + 1, 256, 2147483647u) == 16777217u);
    std::cout << "Indexing and 1D coverage passed\n";
}
